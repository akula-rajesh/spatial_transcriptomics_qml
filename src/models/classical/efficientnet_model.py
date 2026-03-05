"""
Classical EfficientNet model for spatial transcriptomics prediction.
"""

import logging
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
try:
    import torchvision.models as models
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    logging.warning("torchvision not available, using simplified EfficientNet")

from src.models.base_model import BaseModel

logger = logging.getLogger(__name__)

class EfficientNetModel(BaseModel):
    """EfficientNet-based model for predicting spatial gene expression."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the EfficientNet model.
        
        Args:
            config: Configuration dictionary for the model
        """
        super().__init__(config)
        
        # Model architecture parameters
        self.backbone_name = self.get_config_value('architecture.backbone', 'efficientnet_b4')
        self.pretrained = self.get_config_value('architecture.pretrained', True)
        self.use_auxiliary_head = self.get_config_value('architecture.use_auxiliary_head', False)
        self.auxiliary_ratio = self.get_config_value('architecture.auxiliary_ratio', 1.0)
        self.dropout_rate = self.get_config_value('architecture.dropout_rate', 0.2)
        
        # Build the model
        self._build_model()
        
        # Move model to device
        self.to(self.device)
        
    def _build_model(self) -> None:
        """Build the EfficientNet model architecture."""
        try:
            if TORCHVISION_AVAILABLE:
                self._build_torchvision_model()
            else:
                self._build_simplified_model()
        except Exception as e:
            self.log_warning(f"Falling back to simplified model: {str(e)}")
            self._build_simplified_model()
            
    def _build_torchvision_model(self) -> None:
        """Build model using torchvision EfficientNet."""
        # Map config names to torchvision names
        backbone_map = {
            'efficientnet_b0': 'efficientnet_b0',
            'efficientnet_b1': 'efficientnet_b1',
            'efficientnet_b2': 'efficientnet_b2',
            'efficientnet_b3': 'efficientnet_b3',
            'efficientnet_b4': 'efficientnet_b4',
            'efficientnet_b5': 'efficientnet_b5',
            'efficientnet_b6': 'efficientnet_b6',
            'efficientnet_b7': 'efficientnet_b7',
        }
        
        if self.backbone_name not in backbone_map:
            raise ValueError(f"Unsupported EfficientNet variant: {self.backbone_name}")
            
        # Create backbone
        backbone_fn = getattr(models, backbone_map[self.backbone_name])
        self.backbone = backbone_fn(pretrained=self.pretrained)
        
        # Get the number of features from the backbone
        if hasattr(self.backbone, 'classifier'):
            in_features = self.backbone.classifier.in_features
            # Replace classifier with identity to get features
            self.backbone.classifier = nn.Identity()
        elif hasattr(self.backbone, 'fc'):
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            # For some EfficientNet versions, need to access differently
            in_features = self.backbone._fc.in_features
            self.backbone._fc = nn.Identity()
            
        self.features_dim = in_features
        
        # Main prediction head
        self.dropout = nn.Dropout(self.dropout_rate)
        self.main_predictor = nn.Sequential(
            nn.Linear(self.features_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(512, self.output_genes)
        )
        
        # Auxiliary head if enabled
        if self.use_auxiliary_head:
            self.auxiliary_predictor = nn.Sequential(
                nn.Linear(self.features_dim, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout_rate),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout_rate),
                nn.Linear(256, self.get_config_value('output.auxiliary_genes', self.output_genes))
            )
            
        self.log_info(f"Built EfficientNet model with {self.backbone_name}")
        
    def _build_simplified_model(self) -> None:
        """Build a simplified CNN model when torchvision is not available."""
        self.log_info("Building simplified CNN model (torchvision not available)")
        
        # Simple CNN backbone
        self.backbone = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(self.input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Conv Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Conv Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Conv Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # Calculate output size
        self.features_dim = 256
        
        # Main prediction head
        self.dropout = nn.Dropout(self.dropout_rate)
        self.main_predictor = nn.Sequential(
            nn.Linear(self.features_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(512, self.output_genes)
        )
        
        # Auxiliary head if enabled
        if self.use_auxiliary_head:
            self.auxiliary_predictor = nn.Sequential(
                nn.Linear(self.features_dim, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout_rate),
                nn.Linear(256, self.get_config_value('output.auxiliary_genes', self.output_genes))
            )
            
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Tuple of (main_output, auxiliary_output) where auxiliary_output
            may be None if auxiliary head is disabled
        """
        # Extract features using backbone
        features = self.backbone(x)
        
        # Apply dropout
        features = self.dropout(features)
        
        # Main prediction
        main_output = self.main_predictor(features)
        
        # Auxiliary prediction if enabled
        auxiliary_output = None
        if self.use_auxiliary_head:
            auxiliary_output = self.auxiliary_predictor(features)
            
        return main_output, auxiliary_output
        
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
            
    def get_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get features from the backbone only.
        
        Args:
            x: Input tensor
            
        Returns:
            Backbone features
        """
        with torch.no_grad():
            return self.backbone(x)

# For backward compatibility
EfficientNet = EfficientNetModel

__all__ = ['EfficientNetModel', 'EfficientNet']
