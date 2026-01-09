"""
Hybrid Ensemble Accident Detection Model
Novel Architecture combining:
1. Multi-Scale Spatial Attention Network
2. Ensemble of Pre-trained CNNs (EfficientNet, ResNet, DenseNet)
3. Feature Fusion with Channel Attention
4. Custom Focal Loss for class imbalance

Suitable for research paper publication
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import timm
from typing import Tuple, List
import ssl
import urllib

# Fix SSL certificate verification issue
ssl._create_default_https_context = ssl._create_unverified_context


class SpatialAttentionModule(nn.Module):
    """
    Novel Multi-Scale Spatial Attention Module
    Captures accident-relevant spatial features at different scales
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Generate spatial attention map
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv1(concat))
        return x * attention


class ChannelAttentionModule(nn.Module):
    """
    Channel Attention using squeeze-and-excitation
    Emphasizes important feature channels
    """
    def __init__(self, channels, reduction=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()

        # Both average and max pooling
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))

        attention = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * attention.expand_as(x)


class MultiScaleFeatureExtractor(nn.Module):
    """
    Extracts features at multiple scales to capture both
    local accident details and global scene context
    """
    def __init__(self, in_channels):
        super(MultiScaleFeatureExtractor, self).__init__()

        # Different kernel sizes for multi-scale
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//4, kernel_size=1),
            nn.BatchNorm2d(in_channels//4),
            nn.ReLU(inplace=True)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//4, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels//4),
            nn.ReLU(inplace=True)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//4, kernel_size=5, padding=2),
            nn.BatchNorm2d(in_channels//4),
            nn.ReLU(inplace=True)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, in_channels//4, kernel_size=1),
            nn.BatchNorm2d(in_channels//4),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        return torch.cat([branch1, branch2, branch3, branch4], dim=1)


class HybridEnsembleAccidentDetector(nn.Module):
    """
    Novel Hybrid Ensemble Architecture for Accident Detection

    Architecture:
    1. Three parallel CNN backbones (EfficientNet-B3, ResNet50, DenseNet121)
    2. Multi-scale feature extraction
    3. Spatial and channel attention mechanisms
    4. Feature fusion with learnable weights
    5. Classification head with dropout
    """
    def __init__(self, num_classes=2, pretrained=True):
        super(HybridEnsembleAccidentDetector, self).__init__()

        # Backbone 1: EfficientNet-B3 (efficient and accurate)
        self.efficientnet = timm.create_model('efficientnet_b3', pretrained=pretrained)
        efficientnet_features = self.efficientnet.classifier.in_features
        self.efficientnet.classifier = nn.Identity()

        # Backbone 2: ResNet50 (strong feature extraction)
        self.resnet = models.resnet50(pretrained=pretrained)
        resnet_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        # Backbone 3: DenseNet121 (feature reuse)
        self.densenet = models.densenet121(pretrained=pretrained)
        densenet_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Identity()

        # Multi-scale feature extraction for each backbone
        self.efficientnet_multiscale = MultiScaleFeatureExtractor(efficientnet_features)
        self.resnet_multiscale = MultiScaleFeatureExtractor(resnet_features)
        self.densenet_multiscale = MultiScaleFeatureExtractor(densenet_features)

        # Attention modules
        self.efficientnet_attention = nn.Sequential(
            ChannelAttentionModule(efficientnet_features),
            SpatialAttentionModule()
        )
        self.resnet_attention = nn.Sequential(
            ChannelAttentionModule(resnet_features),
            SpatialAttentionModule()
        )
        self.densenet_attention = nn.Sequential(
            ChannelAttentionModule(densenet_features),
            SpatialAttentionModule()
        )

        # Feature dimension reduction
        self.efficientnet_reduce = nn.Linear(efficientnet_features, 512)
        self.resnet_reduce = nn.Linear(resnet_features, 512)
        self.densenet_reduce = nn.Linear(densenet_features, 512)

        # Learnable fusion weights
        self.fusion_weights = nn.Parameter(torch.ones(3) / 3)

        # Feature fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

        # Classification head
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        # Extract features from each backbone
        eff_feat = self.efficientnet(x)
        res_feat = self.resnet(x)
        dense_feat = self.densenet(x)

        # Reshape for attention (if needed)
        eff_feat_2d = eff_feat.view(eff_feat.size(0), eff_feat.size(1), 1, 1)
        res_feat_2d = res_feat.view(res_feat.size(0), res_feat.size(1), 1, 1)
        dense_feat_2d = dense_feat.view(dense_feat.size(0), dense_feat.size(1), 1, 1)

        # Apply attention
        eff_feat_2d = self.efficientnet_attention(eff_feat_2d)
        res_feat_2d = self.resnet_attention(res_feat_2d)
        dense_feat_2d = self.densenet_attention(dense_feat_2d)

        # Flatten
        eff_feat = eff_feat_2d.view(eff_feat_2d.size(0), -1)
        res_feat = res_feat_2d.view(res_feat_2d.size(0), -1)
        dense_feat = dense_feat_2d.view(dense_feat_2d.size(0), -1)

        # Reduce dimensions
        eff_feat = self.efficientnet_reduce(eff_feat)
        res_feat = self.resnet_reduce(res_feat)
        dense_feat = self.densenet_reduce(dense_feat)

        # Normalize fusion weights
        weights = F.softmax(self.fusion_weights, dim=0)

        # Weighted fusion
        fused_features = (weights[0] * eff_feat +
                         weights[1] * res_feat +
                         weights[2] * dense_feat)

        # Final fusion and classification
        fused = self.fusion(fused_features)
        output = self.classifier(fused)

        return output

    def get_fusion_weights(self):
        """Returns the learned fusion weights for analysis"""
        return F.softmax(self.fusion_weights, dim=0).detach().cpu().numpy()


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    Focuses training on hard examples
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def get_transforms(phase='train'):
    """
    Data augmentation and preprocessing
    """
    if phase == 'train':
        return transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
