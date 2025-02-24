import os
from glob import glob
from tqdm import tqdm
import numpy as np
from .utils import create_label_mask
from PIL import Image
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torchvision.transforms.functional as TF
import cv2
import torch.nn.functional as F

def create_gt_images(train_folder_path: str, class_dict: dict) -> None:
    """
    Create ground truth mask images from labeled training data.

    This function processes labeled images (.lab.png) from the training folder and
    converts them into ground truth masks saved as numpy arrays (.npy).

    Args:
        train_folder_path (str): Path to the folder containing training images
        class_dict (dict): Dictionary mapping class labels to their corresponding values

    Returns:
        None: Saves the ground truth masks to disk with '_mask_gt.npy' suffix

    Example:
        >>> class_dict = {'background': 0, 'rock': 1, 'void': 2}
        >>> create_gt_images('./data/train', class_dict)
    """
    # Ensure the training folder exists
    if not os.path.exists(train_folder_path):
        raise FileNotFoundError(f"Training folder not found at: {train_folder_path}")

    # Get all labeled image files
    lab_files = glob(os.path.join(train_folder_path, "*lab.png"))
    print(f"Processing {len(lab_files)} labeled images...")

    # Process each labeled image
    for lab_file in tqdm(sorted(lab_files)):
        mask_gt = create_label_mask(lab_file, class_dict)
        
        # Generate output path and save the ground truth mask
        mask_gt_path = lab_file.replace("lab.png", "mask_gt.npy")
        np.save(mask_gt_path, mask_gt)

    print("Successfully created all ground truth masks") 

def explore_labeled_shapes(train_folder_path: str) -> Tuple[List[dict], Dict[str, float]]:
    """
    Analyze and visualize image dimensions from the training dataset.

    This function explores the dimensions of labeled images in the dataset,
    calculating statistics and generating distribution plots of image sizes.

    Args:
        train_folder_path (str): Path to the directory containing training images

    Returns:
        Tuple[List[dict], Dict[str, float]]: Returns a tuple containing:
            - List of dictionaries with image paths and their dimensions
            - Dictionary with statistical metrics (mean, std, median, max dimensions)

    Raises:
        FileNotFoundError: If the training folder path doesn't exist
        ValueError: If no images are found in the specified path

    Example:
        >>> dimensions, metrics = explore_labeled_shapes('./data/train')
        >>> print(metrics['mean_width'])
    """
    # Validate input path
    if not os.path.exists(train_folder_path):
        raise FileNotFoundError(f"Training folder not found at: {train_folder_path}")

    # Get all labeled images
    labeled_images = glob(os.path.join(train_folder_path, "*lab.png"))
    if not labeled_images:
        raise ValueError(f"No labeled images found in {train_folder_path}")

    # Initialize lists for dimensions
    dimensions = []
    widths = []
    heights = []
    
    # Collect image dimensions
    print(f"Exploring {len(labeled_images)} labeled images...")
    for img_path in tqdm(sorted(labeled_images)):
        with Image.open(img_path) as img:
            width, height = img.size
            dimensions.append({
                'path': img_path,
                'width': width,
                'height': height
            })
            widths.append(width)
            heights.append(height)

    # Convert to numpy arrays for efficient computation
    widths = np.array(widths)
    heights = np.array(heights)
    
    # Calculate statistics
    metrics = {
        'max_width': float(np.max(widths)),
        'max_height': float(np.max(heights)),
        'mean_width': float(np.mean(widths)),
        'mean_height': float(np.mean(heights)),
        'std_width': float(np.std(widths)),
        'std_height': float(np.std(heights)),
        'median_width': float(np.median(widths)),
        'median_height': float(np.median(heights))
    }

    # Print summary statistics
    print("\nImage Dimension Statistics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.2f}")

    # Plot distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.hist(widths, bins=50)
    ax1.set_xlabel('Width (pixels)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Width Distribution')
    
    ax2.hist(heights, bins=50)
    ax2.set_xlabel('Height (pixels)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Height Distribution')
    
    plt.tight_layout()
    plt.show()

    return dimensions, metrics

def analyze_class_distribution(train_folder_path: str) -> Dict[int, int]:
    """
    Analyze the distribution of classes in ground truth masks.

    This function reads all .npy ground truth files and creates a histogram
    of class frequencies across the dataset.

    Args:
        train_folder_path (str): Path to the directory containing ground truth .npy files

    Returns:
        Dict[int, int]: Dictionary mapping class indices to their frequencies

    Raises:
        FileNotFoundError: If the training folder path doesn't exist
        ValueError: If no .npy files are found in the specified path

    Example:
        >>> class_dist = analyze_class_distribution('./data/train')
        >>> print(f"Class 1 appears {class_dist[1]} times")
    """
    # Validate input path
    if not os.path.exists(train_folder_path):
        raise FileNotFoundError(f"Training folder not found at: {train_folder_path}")

    # Get all .npy files
    gt_files = glob(os.path.join(train_folder_path, "*mask_gt.npy"))
    if not gt_files:
        raise ValueError(f"No ground truth .npy files found in {train_folder_path}")

    # Initialize class frequencies dictionary
    class_frequencies = {i: 0 for i in range(9)}  # 9 classes (0-8)
    
    print(f"Processing {len(gt_files)} ground truth files...")
    
    # Process each ground truth file
    for gt_file in tqdm(sorted(gt_files)):
        mask = np.load(gt_file)
        unique, counts = np.unique(mask, return_counts=True)
        for class_idx, count in zip(unique, counts):
            class_frequencies[class_idx-1] += count

    # Plot distribution
    plt.figure(figsize=(12, 6))
    classes = list(class_frequencies.keys())
    frequencies = list(class_frequencies.values())
    
    plt.bar(classes, frequencies)
    plt.title('Class Distribution in Ground Truth Masks')
    plt.xlabel('Class Index')
    plt.ylabel('Frequency (pixel count)')
    plt.xticks(classes)
    
    # Add value labels on top of each bar
    for i, freq in enumerate(frequencies):
        plt.text(i, freq, f'{freq:,}', 
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

    # Print summary
    print("\nClass Distribution Summary:")
    total_pixels = sum(frequencies)
    for class_idx, freq in class_frequencies.items():
        percentage = (freq / total_pixels) * 100
        print(f"Class {class_idx}: {freq:,} pixels ({percentage:.2f}%)")

    return class_frequencies


class DiceLoss(nn.Module):
    """
    Dice Loss for semantic segmentation tasks.
    
    Implements the Dice Loss function with optional smoothing factor to prevent
    division by zero. The loss is calculated per class and averaged.
    """
    
    def __init__(self, smooth: float = 1e-6) -> None:
        """
        Initialize Dice Loss.

        Args:
            smooth: Smoothing factor to prevent division by zero
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate Dice Loss between predictions and targets.

        Args:
            pred: Predicted logits of shape (B, C, H, W)
            target: Ground truth labels of shape (B, H, W)

        Returns:
            Scalar loss value
        """
        # Convert target to one-hot encoding
        n_classes = pred.size(1)
        target_one_hot = F.one_hot(target, n_classes).permute(0, 3, 1, 2).float()
        
        # Apply softmax to predictions
        pred_softmax = F.softmax(pred, dim=1)
        
        # Calculate Dice score for each class
        numerator = 2 * (pred_softmax * target_one_hot).sum(dim=(2, 3))
        denominator = (pred_softmax.pow(2) + target_one_hot.pow(2)).sum(dim=(2, 3))
        dice_score = (numerator + self.smooth) / (denominator + self.smooth)
        
        # Average over classes and batches
        dice_loss = 1 - dice_score.mean()
        return dice_loss

class CombinedLoss(nn.Module):
    """
    Combined Cross Entropy and Dice Loss with class weights support.
    
    This loss function combines Cross Entropy and Dice Loss with a weighted average.
    Optionally supports class weights for handling class imbalance.
    """
    
    def __init__(
        self, 
        alpha: float = 0.5,
        use_weight: bool = False,
        label_smoothing: float = 0.2
    ) -> None:
        """
        Initialize Combined Loss.

        Args:
            alpha: Weight factor for Cross Entropy Loss (1-alpha for Dice Loss)
            use_weight: Whether to use class weights for Cross Entropy Loss
            label_smoothing: Label smoothing factor for Cross Entropy Loss
        """
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        
        # Class weights calculated from dataset distribution
        weights = None
        if use_weight:
            weights = torch.tensor([
                0.1128, 0.0485, 0.1150, 0.1616, 0.0353,
                0.0485, 0.1384, 0.1008, 0.2390
            ]).bfloat16()
        
        # Initialize loss components
        self.ce = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            weight=weights
        )
        self.dice = DiceLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate combined loss.

        Args:
            pred: Predicted logits of shape (B, C, H, W)
            target: Ground truth labels of shape (B, H, W)

        Returns:
            Combined weighted loss value
        """
        ce_loss = self.alpha * self.ce(pred, target)
        dice_loss = (1 - self.alpha) * self.dice(pred, target)
        return ce_loss + dice_loss

class ProbOhemCrossEntropy2d(nn.Module):
    """
    Online Hard Example Mining (OHEM) Cross Entropy Loss for 2D semantic segmentation.
    
    This loss function implements probability-based OHEM, selecting hard examples based on 
    their prediction probabilities. It helps focus training on difficult samples by mining
    hard examples online during the forward pass.
    """
    
    def __init__(
        self,
        ignore_index: int = -1,
        reduction: str = 'mean',
        thresh: float = 0.7,
        min_kept: int = 256,
        use_weight: bool = False,
        label_smoothing: float = 0.2
    ) -> None:
        """
        Initialize OHEM Cross Entropy Loss.

        Args:
            ignore_index: Label value to ignore in loss calculation
            reduction: Reduction method ('mean', 'sum', 'none')
            thresh: Probability threshold for hard example selection
            min_kept: Minimum number of pixels to keep
            use_weight: Whether to use class weights
            label_smoothing: Label smoothing factor
        """
        super(ProbOhemCrossEntropy2d, self).__init__()
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        
        # Class weights calculated from dataset distribution
        if use_weight:
            weight = torch.tensor([
                0.1128, 0.0485, 0.1150, 0.1616, 0.0353,
                0.0485, 0.1384, 0.1008, 0.2390
            ]).float()
        else:
            weight = None
        
        # Initialize cross entropy loss
        self.criterion = nn.CrossEntropyLoss(
            reduction=reduction,
            weight=weight,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate OHEM-based cross entropy loss.

        Args:
            pred: Predicted logits of shape (B, C, H, W)
            target: Ground truth labels of shape (B, H, W)

        Returns:
            Loss value after hard example mining
        """
        # Get input dimensions
        batch_size, num_classes, height, width = pred.size()
        
        # Flatten target and create valid mask
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_index)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        # Calculate probabilities
        prob = F.softmax(pred, dim=1)
        prob = prob.transpose(0, 1).reshape(num_classes, -1)

        # Apply OHEM if we have enough valid pixels
        if self.min_kept < num_valid:
            # Mask invalid positions
            prob = prob.masked_fill_(~valid_mask, 1)
            
            # Get prediction probabilities for ground truth classes
            mask_prob = prob[target, torch.arange(len(target), dtype=torch.long)]
            
            # Determine threshold for hard example selection
            threshold = self.thresh
            if self.min_kept > 0:
                # Sort probabilities and adjust threshold if needed
                index = mask_prob.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                
                # Keep only hard examples
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask

        # Apply valid mask and reshape target
        target = target.masked_fill_(~valid_mask, self.ignore_index)
        target = target.view(batch_size, height, width)

        return self.criterion(pred, target)


def create_plot_sam(image, output,output_sam, class_dict, img_name,masks,img_sam):
    # Get confidence
    conf = torch.nn.functional.softmax(output, dim=1)
    # Get max confidence
    conf_max = torch.max(conf, dim=1)[0].float()
    # Get the predicted class labels
    predicted_labels = output.argmax(dim=1)
    # Convert the predicted labels to numpy array
    predicted_labels = predicted_labels.cpu().numpy()
    # Get the image and target for visualization
    image_vis = image[0].permute(1, 2, 0).cpu().numpy()
    # Denormalize the image
    mean = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(0).unsqueeze(1).numpy()
    std = torch.tensor([0.229, 0.224, 0.225]).unsqueeze(0).unsqueeze(1).numpy()
    image_vis = image_vis * std + mean
    class_dict_arr = np.array([x for x in class_dict.values()])
    # Create a figure with subplots
    fig, axs = plt.subplots(1, 6, figsize=(15, 8))
    # Plot the original image
    axs[0].imshow(image_vis)
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    # Plot the equalized image
    axs[1].imshow(TF.equalize(TF.to_pil_image(image_vis)))
    axs[1].set_title('Equalized Image')
    axs[1].axis('off')
    # # Plot the predicted segmentation mask
    axs[2].imshow(class_dict_arr[predicted_labels[0]])
    axs[2].set_title('Pred Segm')
    axs[2].axis('off')
    # # Plot the confidence
    axs[3].imshow(conf_max[0].cpu().numpy())
    axs[3].set_title('Conf')
    axs[3].axis('off')
    # also add entropy of the prediction
    axs[3].set_title('Conf\nMean conf: {:.4f}\nMedian conf: {:.4f}, \nEntropy: {:.4f}'.format(conf_max.mean().item(), torch.median(conf_max).item(), -torch.sum(conf*torch.log(conf+1e-10), dim=1).mean().item()))
    # Output SAM
    axs[4].imshow(class_dict_arr[output_sam-1])
    axs[4].set_title('Refined Segm SAM')
    axs[4].axis('off')
    # Sam Masks
    # Iterate over the masks and plot them in the same figure using polygons
    image_copy = np.copy(img_sam)
    for mask in masks:
        # Convert mask segmentation to binary mask
        binary_mask = (mask['segmentation'] * 255).astype(np.uint8)
        # Find contours from the binary mask
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Draw contours on the image copy in green color with thickness 2
        cv2.drawContours(image_copy, contours, -1, (0, 255, 0), 2)
    axs[5].imshow(image_copy)
    axs[5].set_title('SAM Masks')
    axs[5].axis('off')
    # # # Show the plot
    plt.show()
    os.makedirs('pred_vis', exist_ok=True)
    plt.savefig(f'pred_vis/{img_name}')
    plt.close()