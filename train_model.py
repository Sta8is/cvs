from src import *
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
import random
import warnings
import torchvision.transforms.functional as TF
import cv2
import argparse



def train_one_epoch(model, train_loader, optimizer, device, epoch, scheduler, accumulation_steps=1, mixed_precision=False):
    # Define loss function
    criterion = nn.CrossEntropyLoss(label_smoothing=0.2)
    model.train()
    total_loss = 0
    # Create a tqdm progress bar
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} Training')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        if mixed_precision:
            data = data.bfloat16()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target.squeeze(1)-1)
        loss = loss / accumulation_steps
        total_loss += loss.item()
        loss.backward()
        # Update weights after accumulating gradients
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'})

    return total_loss / len(train_loader)
    
def evaluate(model, val_loader, device, mixed_precision=False):
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0
    dice_scores = []
    # Create a tqdm progress bar
    pbar = tqdm(val_loader, desc='Validation')
    with torch.no_grad():
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            if mixed_precision:
                data = data.bfloat16()
            output = model(data)
            loss = criterion(output, target.squeeze(1)-1)
            total_loss += loss.item()
            # Update IoU metric
            output_np = output.argmax(1).squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)+1
            dice = calculate_dice(target.squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8), output_np)
            dice_scores.append(dice)

            # Update the progress bar with the current loss
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

    # Compute final dice score
    dice_score = np.nanmean(dice_scores)
    return total_loss / len(val_loader), dice_score






parser = argparse.ArgumentParser()
parser.add_argument('--image_size', type=lambda x: tuple(map(int, x.split(','))), default=(1344,364),
                    help='Image size as height,width')
parser.add_argument('--train_dir', type=str, default='data/train',
                    help='Directory containing training data')
parser.add_argument('--split_train_val', action='store_true', default=False, help='Split training data into train (300 images) and validation (40 images)')
parser.add_argument('--seed', type=int, default=1234, help='Random seed for reproducibility')
parser.add_argument('--batch_size_train', type=int, default=16, help='Training batch size')
parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
parser.add_argument('--accum_steps', type=int, default=1, help='Number of steps for gradient accumulation')
parser.add_argument('--mixed_precision', action='store_true', default=False, help='Use bfloat16 precision')
args = parser.parse_args()

image_size = args.image_size
labeled_images = sorted(glob(os.path.join(args.train_dir, '*img.png')))
# Randomly select 40 images for validation and the rest for training
np.random.seed(0)
np.random.shuffle(labeled_images)
if args.split_train_val:
    train_images = labeled_images[40:]
    val_images = labeled_images[:40]
else:
    train_images = labeled_images
    val_images = labeled_images
print(f'Training images: {len(train_images)}')
print(f'Validation images: {len(val_images)}')

np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create datasets
train_dataset = CoreValuesLabeled(train_images, split='train', image_size=image_size)
val_dataset = CoreValuesLabeled(val_images, split='val', image_size=image_size)
# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=args.batch_size_train, shuffle=True, pin_memory=True, num_workers=16, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4)

# Initialize the model
model = SemanticSegmentationModel(num_classes=9, img_size=image_size, frozen_backbone=True, model_version='vitb').to(device)

# Define optimizer
optimizer = optim.AdamW(model.head.parameters(), lr=args.lr, weight_decay=0.0001,betas=(0.9, 0.999))    
warmup_epochs = int(args.epochs * 0.1)
accum_steps=args.accum_steps
warmup_lr_scheduler = optim.lr_scheduler.LinearLR(optimizer, total_iters=warmup_epochs*len(train_loader)//accum_steps)
main_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs*len(train_loader)//accum_steps, eta_min=1e-5)
scheduler = optim.lr_scheduler.ChainedScheduler([warmup_lr_scheduler, main_lr_scheduler])

model.to(device)
if args.mixed_precision:
    model.bfloat16()

# Count number of model parameters and number of trainable parameters
total_params = sum(p.numel() for p in model.parameters())
print(f'{total_params:,} total parameters.')
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters.')
train_losses = []
val_losses = []
val_dices = []
# Training and validation loop
best_dice = 0
for epoch in range(args.epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, scheduler, accumulation_steps=accum_steps, mixed_precision=args.mixed_precision)
    val_loss, val_dice = evaluate(model, val_loader, device, mixed_precision=args.mixed_precision)
    # scheduler.step()
    print(f'Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val dice: {val_dice:.4f}, Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_dices.append(val_dice)
    if val_dice > best_dice:
        best_dice = val_dice
        torch.save(model.state_dict(), 'checkpoints/best_model_supervised.pth')  
        print(f'New best dice score: {best_dice:.4f} at epoch {epoch+1}')