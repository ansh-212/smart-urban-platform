"""
Novel Dual-Branch Attention Network (DBAN) for Pothole Detection
Research-oriented approach with:
1. Edge-Enhanced Feature Extraction (EEFE)
2. Dual-Branch Attention Mechanism (Spatial + Texture)
3. Multi-Scale Pyramid Detection
4. Lightweight EfficientNet backbone

Optimized for fast training with high accuracy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


class EdgeEnhancedModule(nn.Module):
    """
    Novel Edge-Enhanced Feature Extraction (EEFE)
    Emphasizes pothole boundaries and texture discontinuities
    """
    def __init__(self, in_channels):
        super(EdgeEnhancedModule, self).__init__()

        # Sobel edge detection filters
        self.edge_conv_x = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.edge_conv_y = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)

        # Initialize with Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

        for i in range(in_channels):
            self.edge_conv_x.weight.data[i, i, :, :] = sobel_x
            self.edge_conv_y.weight.data[i, i, :, :] = sobel_y

        # Freeze edge detection weights
        self.edge_conv_x.weight.requires_grad = False
        self.edge_conv_y.weight.requires_grad = False

        # Edge fusion
        self.edge_fusion = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Original features
        original = x

        # Edge detection
        edge_x = self.edge_conv_x(x)
        edge_y = self.edge_conv_y(x)

        # Combine original + edges
        enhanced = torch.cat([original, edge_x, edge_y], dim=1)
        output = self.edge_fusion(enhanced)

        return output


class DualBranchAttention(nn.Module):
    """
    Novel Dual-Branch Attention Mechanism
    Branch 1: Spatial Attention (where are potholes?)
    Branch 2: Texture Attention (what texture patterns?)
    """
    def __init__(self, channels):
        super(DualBranchAttention, self).__init__()

        # Spatial Attention Branch
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        # Texture Attention Branch
        self.texture_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 16, channels, kernel_size=1),
            nn.Sigmoid()
        )

        # Fusion weights (learnable)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_in = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.spatial_attention(spatial_in)

        # Texture attention
        texture_att = self.texture_attention(x)

        # Dual-branch fusion with learnable weights
        weights_norm = F.softmax(torch.stack([self.alpha, self.beta]), dim=0)
        output = weights_norm[0] * (x * spatial_att) + weights_norm[1] * (x * texture_att)

        return output


class MultiScalePyramid(nn.Module):
    """
    Multi-Scale Pyramid for detecting potholes of various sizes
    """
    def __init__(self, in_channels):
        super(MultiScalePyramid, self).__init__()

        # Different scales
        self.scale1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True)
        )

        self.scale2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True)
        )

        self.scale3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=5, padding=2),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True)
        )

        self.scale4 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=7, padding=3),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True)
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        s1 = self.scale1(x)
        s2 = self.scale2(x)
        s3 = self.scale3(x)
        s4 = self.scale4(x)

        multi_scale = torch.cat([s1, s2, s3, s4], dim=1)
        output = self.fusion(multi_scale)

        return output


class NovelPotholeDetector(nn.Module):
    """
    Novel Dual-Branch Attention Network (DBAN) for Pothole Detection

    Key Innovations:
    1. Edge-Enhanced Feature Extraction (EEFE)
    2. Dual-Branch Attention (Spatial + Texture)
    3. Multi-Scale Pyramid Detection
    4. Lightweight EfficientNet-B0 backbone

    Binary classification: Pothole vs No-Pothole
    """
    def __init__(self, num_classes=2, pretrained=True):
        super(NovelPotholeDetector, self).__init__()

        # Lightweight backbone: EfficientNet-B0
        from torchvision.models import efficientnet_b0
        efficientnet = efficientnet_b0(pretrained=pretrained)

        # Extract feature layers at correct positions
        self.stem = efficientnet.features[:2]      # 1/2 resolution, 16 channels
        self.layer1 = efficientnet.features[2:4]   # 1/4 resolution, 24 channels
        self.layer2 = efficientnet.features[4:6]   # 1/8 resolution, 40 channels

        # Correct channel dimensions for EfficientNet-B0
        self.channels = [16, 24, 40]  # Actual EfficientNet-B0 channels

        # Edge-Enhanced Modules
        self.edge_enhance1 = EdgeEnhancedModule(self.channels[0])
        self.edge_enhance2 = EdgeEnhancedModule(self.channels[1])
        self.edge_enhance3 = EdgeEnhancedModule(self.channels[2])

        # Dual-Branch Attention Modules
        self.dual_attention1 = DualBranchAttention(self.channels[0])
        self.dual_attention2 = DualBranchAttention(self.channels[1])
        self.dual_attention3 = DualBranchAttention(self.channels[2])

        # Multi-Scale Pyramid
        self.multi_scale_pyramid = MultiScalePyramid(self.channels[2])

        # Global context
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.channels[2], 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Multi-scale feature extraction
        f1 = self.stem(x)
        f2 = self.layer1(f1)
        f3 = self.layer2(f2)

        # Edge enhancement
        f1 = self.edge_enhance1(f1)
        f2 = self.edge_enhance2(f2)
        f3 = self.edge_enhance3(f3)

        # Dual-branch attention
        f1 = self.dual_attention1(f1)
        f2 = self.dual_attention2(f2)
        f3 = self.dual_attention3(f3)

        # Multi-scale pyramid on deepest features
        f3 = self.multi_scale_pyramid(f3)

        # Global pooling and classification
        features = self.global_pool(f3)
        output = self.classifier(features)

        return output


def get_transforms(phase='train'):
    """Data augmentation for pothole detection"""
    if phase == 'train':
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
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


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
