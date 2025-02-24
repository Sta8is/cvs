import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
from .dpt import DPTHead  # Assuming DPTHead is in a separate file


class SemanticSegmentationModel(nn.Module):
    """
    Semantic segmentation model using DINOv2 backbone with DPT head.
    
    This model combines a Vision Transformer backbone (DINOv2) with a Dense Prediction
    Transformer (DPT) head for semantic segmentation of geological core images.
    """

    def __init__(
        self, 
        num_classes: int = 9,
        img_size: Tuple[int, int] = (1344, 364),
        frozen_backbone: bool = True,
        model_version: str = 'vitb'
    ) -> None:
        """
        Initialize the semantic segmentation model.

        Args:
            num_classes: Number of output classes
            img_size: Input image dimensions (height, width)
            frozen_backbone: Whether to freeze the backbone parameters
            model_version: DINOv2 model version ('vitb', 'vits', 'vitl', 'vitg')
        """
        super().__init__()
        
        # Model configuration
        self.img_size = img_size
        self.patch_size = 14
        self.patch_h = img_size[0] // self.patch_size
        self.patch_w = img_size[1] // self.patch_size
        self.frozen_backbone = frozen_backbone
        
        # Initialize backbone
        self.backbone = torch.hub.load(
            'facebookresearch/dinov2', 
            f'dinov2_{model_version}14_reg'
        )
        
        # Set backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = not frozen_backbone
        
        # Configure DPT head parameters
        feats, out_chans = 128, [96, 192, 384, 768]
        self.layers = [2, 5, 8, 11]
        
        # Initialize DPT head
        self.head = DPTHead(
            nclass=num_classes,
            in_channels=self.backbone.embed_dim,
            features=feats,
            use_bn=True,
            out_channels=out_chans,
            use_clstoken=False
        )
        
        # Initialize dropout distribution
        self.binomial = torch.distributions.binomial.Binomial(probs=0.5)

    def freeze_backbone(self) -> None:
        """Freeze the backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.frozen_backbone = True

    def unfreeze_backbone(self) -> None:
        """Unfreeze the backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.frozen_backbone = False

    def _apply_comp_drop(
        self, 
        features: Tuple[torch.Tensor, ...],
        dropout_prob: float = 0.5
    ) -> Tuple[torch.Tensor, ...]:
        """
        Apply complementary dropout to features.
        
        Args:
            features: Tuple of feature tensors
            dropout_prob: Dropout probability
            
        Returns:
            Tuple of feature tensors with complementary dropout applied
        """
        bs, dim = features[0].shape[0], features[0].shape[-1]
        
        # Create complementary dropout masks
        dropout_mask1 = self.binomial.sample((bs // 2, dim)).to(features[0].device) * 2.0
        dropout_mask2 = 2.0 - dropout_mask1
        
        # Keep some features unchanged
        num_kept = int(bs // 2 * (1 - dropout_prob))
        kept_indexes = torch.randperm(bs // 2)[:num_kept]
        dropout_mask1[kept_indexes, :] = 1.0
        dropout_mask2[kept_indexes, :] = 1.0
        
        # Combine masks and apply to features
        dropout_mask = torch.cat((dropout_mask1, dropout_mask2))
        return tuple(
            (feature * dropout_mask.unsqueeze(1)).to(feature.dtype) 
            for feature in features
        )

    def forward(
        self, 
        x: torch.Tensor,
        comp_drop: bool = False,
        upsample_output: bool = True
    ) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x: Input tensor of shape (B, C, H, W)
            comp_drop: Whether to apply complementary dropout
            upsample_output: Whether to upsample the output to input size

        Returns:
            Segmentation map of shape (B, num_classes, H, W)
        """
        B, C, H, W = x.shape
        patch_h = H // self.patch_size
        patch_w = W // self.patch_size
        
        # Get backbone features
        with torch.no_grad() if self.frozen_backbone else torch.enable_grad():
            features = self.backbone.get_intermediate_layers(
                x, 
                self.layers,
                return_class_token=False,
                norm=True
            )
        
        # Apply complementary dropout if needed
        if comp_drop:
            features = self._apply_comp_drop(features)
        
        # Generate segmentation map
        out = self.head(features, patch_h, patch_w)
        
        # Upsample if needed
        if upsample_output:
            out = F.interpolate(
                out,
                size=self.img_size,
                mode='bilinear',
                align_corners=False
            )
            
        return out