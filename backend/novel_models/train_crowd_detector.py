"""
Fast Training Script for Lightweight Crowd Density Detection
Optimized for Mac/CPU with comprehensive metrics and visualization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_recall_fscore_support
)
from tqdm import tqdm
import os
import json
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

from lightweight_crowd_detector import (
    LightweightCrowdDetector,
    FocalLoss,
    get_transforms
)


class CrowdDataset(Dataset):
    """Custom dataset for crowd density classification"""
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        # Map density ranges to class labels
        self.class_mapping = {
            '0-1000': 0,
            '1000-2000': 1,
            '2000-3000': 2,
            '3000-4000': 3,
            '4000-5000': 4
        }

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.data_frame.iloc[idx]['image'])
        image = Image.open(img_path).convert('RGB')

        # Get class label
        density_type = self.data_frame.iloc[idx]['type']
        label = self.class_mapping[density_type]

        if self.transform:
            image = self.transform(image)

        return image, label


class FastCrowdTrainer:
    """Fast training pipeline optimized for Mac"""
    def __init__(self, data_dir, csv_file, output_dir='crowd_results'):
        self.data_dir = data_dir
        self.csv_file = csv_file
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = os.path.join(output_dir, f'run_{timestamp}')
        os.makedirs(self.run_dir, exist_ok=True)

        # Metrics storage
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.learning_rates = []

        self.class_names = ['0-1000', '1000-2000', '2000-3000', '3000-4000', '4000-5000']

        print("="*70)
        print("🚀 LIGHTWEIGHT CROWD DENSITY DETECTION MODEL")
        print("="*70)
        print(f"📱 Device: {self.device}")
        print(f"📁 Output: {self.run_dir}")

    def prepare_data(self, batch_size=32, val_split=0.15, test_split=0.15):
        """Prepare data loaders with train/val/test split"""
        print("\n📊 Loading dataset...")

        # Load full dataset
        full_dataset = CrowdDataset(
            csv_file=self.csv_file,
            root_dir=self.data_dir,
            transform=get_transforms('train')
        )

        # Calculate split sizes
        total_size = len(full_dataset)
        test_size = int(test_split * total_size)
        val_size = int(val_split * total_size)
        train_size = total_size - test_size - val_size

        # Split dataset
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )

        # Apply appropriate transforms
        train_dataset.dataset.transform = get_transforms('train', img_size=224)
        val_dataset.dataset.transform = get_transforms('val', img_size=224)
        test_dataset.dataset.transform = get_transforms('test', img_size=224)

        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=False  # num_workers=0 for Mac stability
        )

        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=False
        )

        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=False
        )

        print(f"✅ Training samples: {len(train_dataset)}")
        print(f"✅ Validation samples: {len(val_dataset)}")
        print(f"✅ Test samples: {len(test_dataset)}")
        print(f"✅ Classes: {self.class_names}")

    def initialize_model(self, learning_rate=0.001):
        """Initialize lightweight model"""
        print("\n🏗️ Building lightweight model...")

        self.model = LightweightCrowdDetector(num_classes=5, pretrained=True)
        self.model = self.model.to(self.device)

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"📊 Total parameters: {total_params:,}")
        print(f"📊 Trainable parameters: {trainable_params:,}")

        # Loss function
        self.criterion = FocalLoss(alpha=0.25, gamma=2.0)
        self.aux_criterion = nn.CrossEntropyLoss()

        # Optimizer - Adam for faster convergence
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)

        # Aggressive learning rate scheduler for fast training
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=15, eta_min=1e-6
        )

        print("✅ Model initialized (optimized for fast training)")

    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} [Train]', ncols=100)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            # Forward with auxiliary output
            outputs = self.model(inputs, return_aux=True)

            if isinstance(outputs, tuple):
                main_out, aux_out = outputs
                loss = self.criterion(main_out, labels) + 0.3 * self.aux_criterion(aux_out, labels)
                outputs = main_out
            else:
                loss = self.criterion(outputs, labels)

            # Backward
            loss.backward()
            self.optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({'loss': f'{loss.item():.3f}', 'acc': f'{100.*correct/total:.1f}%'})

        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def validate(self, data_loader, phase='Val'):
        """Validate model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            pbar = tqdm(data_loader, desc=f'{phase}', ncols=100)
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs, return_aux=False)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                pbar.set_postfix({'loss': f'{loss.item():.3f}', 'acc': f'{100.*correct/total:.1f}%'})

        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc, all_preds, all_labels

    def train(self, num_epochs=15, early_stopping_patience=5):
        """Fast training loop"""
        print(f"\n🏋️ Training for {num_epochs} epochs (optimized for speed)...")

        best_val_acc = 0.0
        patience_counter = 0
        start_time = time.time()

        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)

            # Validate
            val_loss, val_acc, _, _ = self.validate(self.val_loader, 'Val')
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)

            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)

            # Print summary
            print(f"\n{'='*70}")
            print(f"Epoch {epoch+1}/{num_epochs} | LR: {current_lr:.6f}")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"{'='*70}")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                }, os.path.join(self.run_dir, 'best_model.pth'))
                print(f"💾 New best model saved! (Val Acc: {val_acc:.2f}%)")
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print(f"\n⚠️ Early stopping at epoch {epoch+1}")
                break

        training_time = time.time() - start_time
        print(f"\n✅ Training completed in {training_time/60:.2f} minutes")
        print(f"🏆 Best Val Acc: {best_val_acc:.2f}%")

        # Save history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'learning_rates': self.learning_rates,
            'best_val_acc': best_val_acc,
            'training_time_minutes': training_time/60
        }
        with open(os.path.join(self.run_dir, 'history.json'), 'w') as f:
            json.dump(history, f, indent=4)

        return history

    def evaluate_test_set(self):
        """Comprehensive test evaluation"""
        print("\n🧪 Evaluating on test set...")

        # Load best model
        checkpoint = torch.load(os.path.join(self.run_dir, 'best_model.pth'))
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Evaluate
        test_loss, test_acc, all_preds, all_labels = self.validate(self.test_loader, 'Test')

        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )

        print("\n" + "="*70)
        print("📊 TEST SET RESULTS")
        print("="*70)
        print(f"Accuracy:  {test_acc:.2f}%")
        print(f"Precision: {precision*100:.2f}%")
        print(f"Recall:    {recall*100:.2f}%")
        print(f"F1-Score:  {f1*100:.2f}%")
        print("="*70)

        # Detailed report
        print("\n" + "="*70)
        print("📊 DETAILED CLASSIFICATION REPORT")
        print("="*70)
        print(classification_report(all_labels, all_preds, target_names=self.class_names, zero_division=0))

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)

        # Save metrics
        metrics = {
            'test_accuracy': test_acc,
            'test_loss': test_loss,
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        }
        with open(os.path.join(self.run_dir, 'test_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)

        return metrics, cm, all_preds, all_labels

    def plot_all(self, cm, all_preds, all_labels):
        """Generate all visualizations"""
        print("\n📊 Generating visualizations...")

        # 1. Training History
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Loss
        axes[0, 0].plot(self.train_losses, label='Train', linewidth=2, color='#3498db')
        axes[0, 0].plot(self.val_losses, label='Val', linewidth=2, color='#e74c3c')
        axes[0, 0].set_title('Loss Curves', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        # Accuracy
        axes[0, 1].plot(self.train_accs, label='Train', linewidth=2, color='#3498db')
        axes[0, 1].plot(self.val_accs, label='Val', linewidth=2, color='#e74c3c')
        axes[0, 1].set_title('Accuracy Curves', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

        # Learning Rate
        axes[1, 0].plot(self.learning_rates, linewidth=2, color='#2ecc71')
        axes[1, 0].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(alpha=0.3)

        # Train-Val Gap
        gap = np.array(self.train_accs) - np.array(self.val_accs)
        axes[1, 1].plot(gap, linewidth=2, color='#9b59b6')
        axes[1, 1].axhline(0, color='black', linestyle='--', alpha=0.3)
        axes[1, 1].set_title('Generalization Gap', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Train - Val Accuracy (%)')
        axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        print("✅ Saved: training_curves.png")
        plt.close()

        # 2. Confusion Matrix
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   ax=axes[0], cbar_kws={'label': 'Count'})
        axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Predicted Density')
        axes[0].set_ylabel('True Density')

        # Normalized
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   ax=axes[1], cbar_kws={'label': 'Percentage'})
        axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Predicted Density')
        axes[1].set_ylabel('True Density')

        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        print("✅ Saved: confusion_matrix.png")
        plt.close()

        # 3. Per-class Performance
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_preds, zero_division=0
        )

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(self.class_names))
        width = 0.25

        ax.bar(x - width, precision * 100, width, label='Precision', color='#3498db')
        ax.bar(x, recall * 100, width, label='Recall', color='#e74c3c')
        ax.bar(x + width, f1 * 100, width, label='F1-Score', color='#2ecc71')

        ax.set_xlabel('Crowd Density Class', fontsize=12)
        ax.set_ylabel('Score (%)', fontsize=12)
        ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names)
        ax.legend()
        ax.grid(alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, 'per_class_metrics.png'), dpi=300, bbox_inches='tight')
        print("✅ Saved: per_class_metrics.png")
        plt.close()

        print(f"\n✅ All visualizations saved to: {self.run_dir}")


def main():
    """Main training pipeline"""
    print("\n" + "="*70)
    print("🌟 LIGHTWEIGHT CROWD DENSITY DETECTION")
    print("Novel Multi-Scale Approach for Research Paper")
    print("="*70)

    # Configuration (optimized for FAST training on Mac)
    DATA_DIR = '/Users/ojasbayas/PycharmProjects/Smart City/novel_models/crowd_dataset'
    CSV_FILE = os.path.join(DATA_DIR, 'crowds_counting.csv')
    OUTPUT_DIR = '/Users/ojasbayas/PycharmProjects/Smart City/novel_models/crowd_results'

    BATCH_SIZE = 32  # Good balance for Mac
    NUM_EPOCHS = 15  # Fast training
    LEARNING_RATE = 0.001

    # Initialize trainer
    trainer = FastCrowdTrainer(DATA_DIR, CSV_FILE, OUTPUT_DIR)

    # Prepare data
    trainer.prepare_data(batch_size=BATCH_SIZE, val_split=0.15, test_split=0.15)

    # Initialize model
    trainer.initialize_model(learning_rate=LEARNING_RATE)

    # Train
    history = trainer.train(num_epochs=NUM_EPOCHS, early_stopping_patience=5)

    # Evaluate
    metrics, cm, all_preds, all_labels = trainer.evaluate_test_set()

    # Generate visualizations
    trainer.plot_all(cm, all_preds, all_labels)

    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print(f"📁 Results: {trainer.run_dir}")
    print("="*70)


if __name__ == '__main__':
    main()

