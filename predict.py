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
import sys
import argparse
sys.path.append('sam2')
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

class_dict = {
    1: (0, 77, 38),
    2: (255, 255, 255),
    3: (203, 20, 20),
    4: (191, 191, 147),
    5: (133, 223, 246),
    6: (159, 157, 12),
    7: (230, 193, 156),
    8: (139, 87, 42),
    9: (200, 200, 200),
}

def predict_tta( model: torch.nn.Module, img: torch.Tensor) -> torch.Tensor:
    """
    Perform Test Time Augmentation (TTA) with multiple flip transformations.
    
    Args:
        model: Neural network model for inference
        img: Input tensor of shape (B, C, H, W)
    Returns:
        Averaged prediction tensor
    """
    all_predictions = []
    output = model(img)
    all_predictions.append(output)
    # Horizontal flip prediction
    data_hflipped = TF.hflip(img)
    output_hflipped = model(data_hflipped)
    output_hflipped = TF.hflip(output_hflipped)
    all_predictions.append(output_hflipped)
    # Vertical flip prediction
    data_vflipped = TF.vflip(img)
    output_vflipped = model(data_vflipped)
    output_vflipped = TF.vflip(output_vflipped)
    all_predictions.append(output_vflipped)
    # Horizontal and vertical flip prediction
    data_flipped = TF.hflip(data_vflipped)
    output_flipped = model(data_flipped)
    output_flipped = TF.hflip(output_flipped)
    output_flipped = TF.vflip(output_flipped)
    all_predictions.append(output_flipped)
    # Average the predictions
    output = torch.stack(all_predictions).mean(0)
    return output

parser = argparse.ArgumentParser()
parser.add_argument('--image_size', type=lambda x: tuple(map(int, x.split(','))), default=(1344,364),help='Image size as height,width')
parser.add_argument('--test_dir', type=str, default='data/core-values-test-data',help='Directory containing training data')
parser.add_argument('--checkpoints_folder', type=str, default='final_checkpoints',help='Folder containing model checkpoints')
parser.add_argument('--sam2_abspath', type=str, default='sam2',help='Absolute path to sam2 checkpoint')
parser.add_argument('--conf_threshold', type=float, default=0.625, help='Confidence threshold for SAM2')
parser.add_argument('--save_visualisations', action='store_true', help='Save visualisations')
args = parser.parse_args()


test_set = sorted(glob(os.path.join(args.test_dir, '*img.png')),key=lambda x: int(os.path.basename(x).split('_')[0]))
print("Number of test images:", len(test_set))
test_transform = T.Compose([
    T.Resize(size=args.image_size),
    T.PILToTensor(),
    T.ToDtype(torch.float, scale=True),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# Load the models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_names = os.listdir(args.checkpoints_folder)
model_paths = [os.path.join(args.checkpoints_folder, model) for model in model_names]
models = []
for model_path in model_paths:
    model = SemanticSegmentationModel(num_classes=9, img_size=args.image_size, frozen_backbone=True, model_version='vitb').to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    model = model.to(device).eval()
    models.append(model)


model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
sam2 = build_sam2(model_cfg, args.sam2_abspath, device)
mask_generator = SAM2AutomaticMaskGenerator(model=sam2,points_per_batch=512)
# Make predictions
pred_masks_dict = {}
for i, test_img in enumerate(tqdm(test_set)):
    img_name = os.path.basename(test_img)
    img_o = Image.open(test_img).convert('RGB')
    img_w, img_h = img_o.size
    img = test_transform(img_o).unsqueeze(0).to(device)
    with torch.no_grad():
        all_predictions = []
        for m, model in enumerate(models):
            output_m = predict_tta(model, img)
            all_predictions.append(output_m)
        # Average the predictions
        output = torch.stack(all_predictions).mean(0)
        # Resize the output to the original image size
        output = F.interpolate(output, size=(img_h, img_w), mode='bilinear', align_corners=False)
        # Convert the output to numpy array (+1 to match the class IDs)
        output_np = (output.argmax(1).squeeze(0).cpu().numpy()+1).astype(np.uint8)
        # Post-process the output 
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        output_np = cv2.morphologyEx(output_np, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(output_np, cv2.MORPH_CLOSE, kernel)
        output_np = mask

        # SAM2 Refinement
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            img_sam = Image.open(test_img)
            img_sam = np.array(img_sam.convert("RGB"))
            # Generate masks
            masks = mask_generator.generate(img_sam)
            masks=sorted(masks, key=(lambda x: x['area']), reverse=True)
            # Refine the masks
            refinedMask = np.ones((masks[0]['segmentation'].shape[0], masks[0]['segmentation'].shape[1]))*255
            # For each mask, find the most frequent class ID based on predicted mask and assign it to the whole area of the mask
            for ann in masks:
                m = ann['segmentation']
                unique, counts = np.unique(output_np[m], return_counts=True)    
                freqs=(np.asarray((unique, counts)).T)
                freqs=freqs[freqs[:, 1].argsort()]
                mostFrequentID=freqs[-1,0]
                refinedMask[m] = mostFrequentID
            
            refined_mask = refinedMask.astype(np.uint8)
            # Sam2 masks do not cover the whole image, so we need to fill the missing parts with the original mask
            non_valid = (refined_mask==255)
            refined_mask[non_valid] = output_np[non_valid]
            output_sam = refined_mask.astype(np.uint8)
            # Replace the refined mask with the original mask if the confidence is higher than the threshold
            conf = torch.nn.functional.softmax(output, dim=1)
            conf_max = torch.max(conf, dim=1)[0].float()
            high_conf_mask = conf_max >= args.conf_threshold
            output_sam[high_conf_mask[0].cpu().numpy()] = output_np[high_conf_mask[0].cpu().numpy()]
            if args.save_visualisations:
                create_plot_sam(img, output, output_sam, class_dict, img_name, masks, img_sam)
            pred_masks_dict[img_name.replace('img','lab').replace(".png","")] = output_sam.astype(np.uint8)
np.savez_compressed('pred_masks.npz', **pred_masks_dict)


