"""
AuxNet model - Enhanced classical model with auxiliary pathways for spatial transcriptomics.
"""

import logging
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base_model import BaseModel

logger = logging.getLogger(__name__)

class AuxNetModel(BaseModel):
    """Auxiliary Network model with multiple pathway integration for spatial gene expression prediction."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the AuxNet model.
        
        Args:
            config: Configuration dictionary for the model
        """
        super().__init__(config)
        
        # Model architecture parameters
        self.backbone_channels = self.get_config_value('architecture.backbone_channels', [32, 64, 128, 256, 512])
        self.auxiliary_branches = self.get_config_value('architecture.auxiliary_branches', 3)
        self.fusion_method = self.get_config_value('architecture.fusion_method', 'concat')
        self.use_spatial_attention = self.get_config_value('architecture.use_spatial_attention', True)
        self.use_channel_attention = self.get_config_value('architecture.use_channel_attention', True)
        
        # Build the model
        self._build_model()
        
        # Move model to device
        self.to(self.device)
        
    def _build_model(self) -> None:
        """Build the AuxNet model architecture."""
        # Main backbone encoder
        self.backbone = self._build_backbone()
        
        # Auxiliary branches
        self.auxiliary_branches_list = nn.ModuleList([
            self._build_auxiliary_branch(i) for i in range(self.auxiliary_branches)
        ])
        
        # Attention mechanisms
        if self.use_spatial_attention:
            self.spatial_attention = SpatialAttention()
            
        if self.use_channel_attention:
            self.channel_attention = ChannelAttention(self.backbone_channels[-1])
        
        # Feature fusion layer
        fusion_features = self._calculate_fusion_features()
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate)
        )
        
        # Main prediction head
        self.main_predictor = nn.Linear(512, self.output_genes)
        
        # Auxiliary prediction heads
        self.auxiliary_predictors = nn.ModuleList([
            nn.Linear(512, min(100, self.output_genes)) 
            for _ in range(self.auxiliary_branches)
        ])
        
        self.log_info(f"Built AuxNet model with {self.auxiliary_branches} auxiliary branches")
        
    def _build_backbone(self) -> nn.Module:
        """Build the main backbone encoder."""
        layers = []
        in_channels = self.input_channels
        
        for i, out_channels in enumerate(self.backbone_channels):
            layers.append(ConvBlock(in_channels, out_channels, 3, padding=1))
            layers.append(nn.MaxPool2d(2))
            in_channels = out_channels
            
        # Add global average pooling
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        layers.append(nn.Flatten())
        
        return nn.Sequential(*layers)
        
    def _build_auxiliary_branch(self, branch_index: int) -> nn.Module:
        """Build an auxiliary branch."""
        # Each branch processes features at different scales
        channels = self.backbone_channels[branch_index % len(self.backbone_channels)]
        
        return nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=1),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
    def _calculate_fusion_features(self) -> int:
        """Calculate the number of features after fusion."""
        backbone_features = self.backbone_channels[-1]
        aux_features = sum([self.backbone_channels[i % len(self.backbone_channels)] // 4 
                           for i in range(self.auxiliary_branches)])
        
        if self.fusion_method == 'concat':
            return backbone_features + aux_features
        else:  # sum or attention
            return backbone_features
            
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Tuple of (main_output, auxiliary_outputs)
        """
        # Extract features through backbone
        features = self._extract_features(x)
        
        # Apply attention if enabled
        if self.use_spatial_attention:
            features = self.spatial_attention(features) * features
            
        if self.use_channel_attention:
            features = self.channel_attention(features) * features
            
        # Get backbone features
        backbone_features = self.backbone[:len(self.backbone)-2](x)  # Before pooling and flatten
        pooled_features = self.backbone[len(self.backbone)-2:](backbone_features)
        
        # Process auxiliary branches
        aux_outputs = []
        aux_features_list = []
        
        # Get intermediate features for auxiliary branches
        for i, branch in enumerate(self.auxiliary_branches_list):
            # Extract features at different scales
            scale_idx = i % (len(self.backbone_channels) - 1)
            if scale_idx < len(self.backbone_channels) - 1:
                # Get intermediate features
                intermediate_features = x
                for j in range(scale_idx + 1):
                    intermediate_features = self.backbone[2*j:2*j+2](intermediate_features)
                    
                aux_features = branch[:-2](intermediate_features)  # Before pooling and flatten
                aux_features = branch[-2:](aux_features)  # Pooling and flatten
                aux_features_list.append(aux_features)
                
                # Auxiliary prediction
                aux_pred = self.auxiliary_predictors[i](self.fusion_layer(aux_features)[:,:512])
                aux_outputs.append(aux_pred)
        
        # Fuse features
        if self.fusion_method == 'concat' and aux_features_list:
            fused_features = torch.cat([pooled_features] + aux_features_list, dim=1)
        else:
            fused_features = pooled_features
            
        # Process through fusion layer
        fused_processed = self.fusion_layer(fused_features)
        
        # Main prediction
        main_output = self.main_predictor(fused_processed)
        
        return main_output, tuple(aux_outputs)
        
    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input tensor."""
        return self.backbone(x)
        
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict gene expression from input images.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Predicted gene expression tensor
        """
        with torch.no_grad():
            main_output, _ = self.forward(x)
            return main_output

class ConvBlock(nn.Module):
    """Convolutional block with conv-batchnorm-relu."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))

class SpatialAttention(nn.Module):
    """Spatial attention mechanism."""
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        x_cat = self.conv1(x_cat)
        return self.sigmoid(x_cat)

class ChannelAttention(nn.Module):
    """Channel attention mechanism."""
    
    def __init__(self, in_planes: int, ratio: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

__all__ = ['AuxNetModel']
