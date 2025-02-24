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


def consistency_loss(pred1, pred2, temperature=0.5):
    """Compute consistency loss between two predictions"""
    pred1 = F.softmax(pred1 / temperature, dim=1)
    pred2 = F.softmax(pred2 / temperature, dim=1)
    return F.mse_loss(pred1, pred2)

def train_one_epoch_semi(model,model_ema, labaled_loader, train_unlabaled_loader, optimizer, device, epoch, scheduler, accumulation_steps=1, mixed_precision=False, loss_function='ce', cons_reg=False):
    # Define loss function
    criterions = {
        'ce': nn.CrossEntropyLoss(label_smoothing=0.2),
        'combined': CombinedLoss(alpha=0.5, use_weight=False),
        'ohem': ProbOhemCrossEntropy2d(ignore_index=255, thresh=0.7, min_kept=200000, use_weight=False)
    }
    criterion_l = criterions[loss_function]
    criterion_u = nn.CrossEntropyLoss(reduction='none')
    model.train()
    total_loss = 0
    conf_thres = 0.7
    
    loader = zip(train_loader, train_unlabaled_loader)

    # Create a tqdm progress bar
    pbar = tqdm(loader, desc=f'Epoch {epoch+1} Training')
    
    
    for batch_idx, ((img_x, mask_x),
            (img_u_w, img_u_s1, img_u_s2, cutmix_box1, cutmix_box2)) in enumerate(pbar):
        img_x, mask_x = img_x.to(device).bfloat16(), mask_x.to(device).long()
        if mixed_precision:
            img_u_w, img_u_s1, img_u_s2 = img_u_w.to(device).bfloat16(), img_u_s1.to(device).bfloat16(), img_u_s2.to(device).bfloat16()
        cutmix_box1, cutmix_box2 = cutmix_box1.to(device), cutmix_box2.to(device)
        # Generate pseudo-labels using teacher model (EMA)
        with torch.no_grad():
            pred_u_w = model_ema(img_u_w).detach()
            conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
            mask_u_w = pred_u_w.argmax(dim=1)
        # Create CutMix versions of pseudo-labels and confidence scores
        # by mixing original predictions with their flipped versions
        img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = img_u_s1.flip(0)[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
        img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = img_u_s2.flip(0)[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]
        # Forward pass through student model
        pred_x = model(img_x)
        # Create CutMix versions of pseudo-labels and confidence scores
        pred_u_s1, pred_u_s2 = model(torch.cat((img_u_s1, img_u_s2)), comp_drop=True).chunk(2)
        mask_u_w_cutmixed1, conf_u_w_cutmixed1 = mask_u_w.clone(), conf_u_w.clone()
        mask_u_w_cutmixed2, conf_u_w_cutmixed2 = mask_u_w.clone(), conf_u_w.clone()
        mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w.flip(0)[cutmix_box1 == 1]
        conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w.flip(0)[cutmix_box1 == 1]
        mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w.flip(0)[cutmix_box2 == 1]
        conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w.flip(0)[cutmix_box2 == 1]
        # Calculate supervised loss for labeled data
        loss_x = criterion_l(pred_x, mask_x.squeeze(1)-1)
        # Calculate unsupervised losses with confidence thresholding
        # Only retain loss for predictions above confidence threshold
        loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed1)
        loss_u_s1 = loss_u_s1 * (conf_u_w_cutmixed1  > conf_thres)
        loss_u_s1 = loss_u_s1.mean()
        loss_u_s2 = criterion_u(pred_u_s2, mask_u_w_cutmixed2)
        loss_u_s2 = loss_u_s2 * (conf_u_w_cutmixed2  > conf_thres)
        loss_u_s2 = loss_u_s2.mean()
        # Consistency regularization
        if cons_reg:
            cons_loss_u_s1 = consistency_loss(pred_u_s1, pred_u_s2)
            cons_loss_ws = consistency_loss(pred_u_w, pred_u_s1)
            cons_loss = (cons_loss_u_s1 + cons_loss_ws)/2
            cons_weight = min(epoch/50, 1)
            loss_u_s = loss_u_s1 + loss_u_s2 + cons_weight * cons_loss
        # Unlabaled Loss
        else:
            loss_u_s = (loss_u_s1 + loss_u_s2) / 2
        # Total Loss
        loss = (loss_x + loss_u_s) / 2
        # loss = (loss_x + loss_u_s + cons_weight * cons_loss)
        loss = loss / accumulation_steps
        total_loss += loss.item()
        loss.backward()
        # Apply gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # Update weights after accumulating gradients
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader): 
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        # Update the progress bar with the current loss and learning rate
        pbar.set_postfix({'Loss': f'{loss.item():.4f}', 
                          'Unlabelled_loss': f'{loss_u_s.item():.4f}',
                          'Labelled_loss': f'{loss_x.item():.4f}',
                          'LR backbone': f'{optimizer.param_groups[0]["lr"]:.6f}', 
                          'LR head': f'{optimizer.param_groups[1]["lr"]:.6f}'})

    iters = epoch * len(train_loader) + batch_idx
    ema_ratio = min(1 - 1 / (iters + 1), 0.996)
    # Update EMA model
    for param, param_ema in zip(model.parameters(), model_ema.parameters()):
        param_ema.copy_(param_ema * ema_ratio + param.detach() * (1 - ema_ratio))
    for buffer, buffer_ema in zip(model.buffers(), model_ema.buffers()):
        buffer_ema.copy_(buffer_ema * ema_ratio + buffer.detach() * (1 - ema_ratio))

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
parser.add_argument('--image_size', type=lambda x: tuple(map(int, x.split(','))), default=(1344,364),help='Image size as height,width')
parser.add_argument('--train_dir', type=str, default='data/train',help='Directory containing training data')
parser.add_argument('--unlabaled_train_dir', type=str, default='data/train_unlabeled',help='Directory containing unlabeled training data')
parser.add_argument('--split_train_val', action='store_true', default=False, help='Split training data into train (300 images) and validation (40 images)')
parser.add_argument('--seed', type=int, default=1234, help='Random seed for reproducibility')
parser.add_argument('--batch_size_train', type=int, default=16, help='Training batch size')
parser.add_argument('--blr', type=float, default=4e-5, help='Learning rate of backbone')
parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate of head')
parser.add_argument('--epochs', type=int, default=150, help='Number of training epochs')
parser.add_argument('--accum_steps', type=int, default=1, help='Number of steps for gradient accumulation')
parser.add_argument('--mixed_precision', action='store_true', default=False, help='Use bfloat16 precision')
parser.add_argument('--loss_function', type=str, default='ce', choices=['ce', 'combined', 'ohem'], help='Loss function to use (ce or dice)')
parser.add_argument('--concistency_reg_unsup', action='store_true', default=False, help='Use consistency regularization for unsupervised data')
parser.add_argument('--sp_model_ckpt', type=str, default='checkpoints/model_supervised.pth', help='Checkpoint path of supervised model to initialize the model')
args = parser.parse_args()

labeled_images = sorted(glob(os.path.join(args.train_dir, '*img.png')))
unlabaled_images = sorted(glob(os.path.join(args.unlabaled_train_dir, '*.png')))
# Randomly select 40 images for validation and the rest for training
np.random.seed(0)
np.random.shuffle(labeled_images)
if args.split_train_val:
    train_images = labeled_images[40:]
    val_images = labeled_images[:40]
else:
    train_images = labeled_images
    val_images = labeled_images
print(f'Training labaled images: {len(train_images)}')
print(f'Training unlabaled images: {len(unlabaled_images)}')
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
train_dataset = CoreValuesSemi(train_images, split='train_labeled', image_size=args.image_size)
train_unlabaled_dataset = CoreValuesSemi(unlabaled_images, split='train_unlabeled', image_size=args.image_size)
val_dataset = CoreValuesSemi(val_images, split='val', image_size=args.image_size)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=args.batch_size_train, shuffle=True, pin_memory=True, num_workers=16, drop_last=True)
train_unlabaled_loader = DataLoader(train_unlabaled_dataset, batch_size=args.batch_size_train, shuffle=True, pin_memory=True, num_workers=16, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4)

# Initialize the model without freezing the backbone and create an EMA model. Also, load the supervised model checkpoint
model = SemanticSegmentationModel(num_classes=9, img_size=args.image_size, frozen_backbone=False, model_version='vitb').to(device)
model.load_state_dict(torch.load(args.sp_model_ckpt, weights_only=True, map_location=device))
model_ema = copy.deepcopy(model)
for param in model_ema.parameters():
    param.requires_grad = False
model_ema = model_ema.to(device)

# Define optimizer
optimizer = optim.AdamW([{'params': model.backbone.parameters(), 'lr': args.blr}, {'params': model.head.parameters(), 'lr': args.lr}], 
                        betas=(0.9, 0.999), weight_decay=0.01)
warmup_epochs = int(args.epochs * 0.1)
accum_steps=args.accum_steps
warmup_lr_scheduler = optim.lr_scheduler.LinearLR(optimizer, total_iters=warmup_epochs*len(train_loader)//accum_steps)
main_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs*len(train_loader)//accum_steps, eta_min=1e-5)
scheduler = optim.lr_scheduler.ChainedScheduler([warmup_lr_scheduler, main_lr_scheduler])

model.to(device)
if args.mixed_precision:
    model.bfloat16()
    model_ema.bfloat16()

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
    train_loss = train_one_epoch_semi(model, model_ema, train_loader, train_unlabaled_loader, 
                                      optimizer, device, epoch, scheduler, 
                                      accumulation_steps=accum_steps, mixed_precision=args.mixed_precision, 
                                      loss_function=args.loss_function, cons_reg=args.concistency_reg_unsup)
    val_loss, val_dice = evaluate(model, val_loader, device, mixed_precision=args.mixed_precision)
    # scheduler.step()
    print(f'Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val dice: {val_dice:.4f}, Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_dices.append(val_dice)
    if val_dice > best_dice:
        best_dice = val_dice
        torch.save(model.state_dict(), 'checkpoints/best_model_semi.pth')  
        print(f'New best dice score: {best_dice:.4f} at epoch {epoch+1}')