"""
Improved ST-GCN Model with fixes based on BiLSTM success patterns
Key changes:
1. Added class balancing with weighted loss
2. Improved correctness classifier architecture
3. Fixed graph structure for REHAB skeleton
4. Added data augmentation
5. Better regularization strategy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from dataset_loader import create_dataloaders
import os

MODEL_DIR = "models"
GRAPH_DIR = "graphs"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)


class GraphConvolution(nn.Module):
    """Spatial Graph Convolution Layer"""
    def __init__(self, in_channels, out_channels, num_subsets=3, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_subsets = num_subsets
        
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels * num_subsets,
            kernel_size=1,
            bias=bias
        )
        
    def forward(self, x, adjacency_matrix):
        N, C, T, V = x.shape
        x = self.conv(x)
        x = x.view(N, self.num_subsets, self.out_channels, T, V)
        x = torch.einsum('nkctv,kvw->nctw', x, adjacency_matrix)
        return x.contiguous()


class TemporalConvolution(nn.Module):
    """Temporal Convolution with Layer Normalization"""
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, dropout=0.3):
        super().__init__()
        padding = (kernel_size - 1) // 2
        
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
            padding=(padding, 0)
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class STGCNBlock(nn.Module):
    """Spatial-Temporal Graph Convolutional Block"""
    def __init__(self, in_channels, out_channels, num_subsets=3, 
                 temporal_kernel_size=9, stride=1, dropout=0.3, residual=True):
        super().__init__()
        
        self.residual = residual
        
        self.gcn = GraphConvolution(in_channels, out_channels, num_subsets)
        self.bn_gcn = nn.BatchNorm2d(out_channels)
        self.tcn = TemporalConvolution(
            out_channels, 
            out_channels,
            kernel_size=temporal_kernel_size,
            stride=stride,
            dropout=dropout
        )
        
        if not residual:
            self.residual_conv = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual_conv = lambda x: x
        else:
            self.residual_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x, adjacency_matrix):
        residual = self.residual_conv(x)
        x = self.gcn(x, adjacency_matrix)
        x = self.bn_gcn(x)
        x = self.relu(x)
        x = self.tcn(x)
        x = x + residual
        x = self.relu(x)
        return x


class ImprovedSTGCN(nn.Module):
    """
    Improved ST-GCN with better correctness detection
    Based on successful BiLSTM patterns
    """
    def __init__(self, num_joints=26, num_coords=2, num_exercises=6, 
                 num_subsets=3, dropout=0.3, adaptive_graph=False):
        super().__init__()
        
        self.num_joints = num_joints
        self.num_coords = num_coords
        self.num_exercises = num_exercises
        
        in_channels = num_coords
        
        # Build fixed adjacency matrix
        self.register_buffer('base_adjacency', self.build_adjacency_matrix())
        
        # Input normalization - use LayerNorm like BiLSTM
        self.input_bn = nn.BatchNorm2d(in_channels)
        
        # Reduced ST-GCN layers to prevent overfitting
        self.stgcn_layers = nn.ModuleList([
            STGCNBlock(in_channels, 64, num_subsets, stride=1, dropout=dropout),
            STGCNBlock(64, 64, num_subsets, stride=1, dropout=dropout),
            STGCNBlock(64, 128, num_subsets, stride=2, dropout=dropout),
            STGCNBlock(128, 128, num_subsets, stride=1, dropout=dropout),
            STGCNBlock(128, 256, num_subsets, stride=2, dropout=dropout),
            STGCNBlock(256, 256, num_subsets, stride=1, dropout=dropout),
        ])
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Shared feature extraction (like BiLSTM)
        self.feature_extractor = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Exercise classifier - simpler (already works well)
        self.exercise_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_exercises)
        )
        
        # Correctness classifier - DEEPER like in BiLSTM (0.8 weight)
        self.correctness_classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1)
        )
        
    def build_adjacency_matrix(self):
        """Build adjacency for REHAB skeleton (26 joints)"""
        num_subsets = 3
        V = self.num_joints
        
        adjacency = torch.zeros(num_subsets, V, V)
        
        # REHAB skeleton connections
        edges = [
            # Torso (0-12)
            (0, 9), (9, 10), (10, 11), (11, 12),
            # Left leg (1, 3, 5, 7)
            (0, 1), (1, 3), (3, 5), (5, 7),
            # Right leg (2, 4, 6, 8)
            (0, 2), (2, 4), (4, 6), (6, 8),
            # Left arm (13, 15, 17, 19)
            (10, 13), (13, 15), (15, 17), (17, 19),
            # Right arm (14, 16, 18, 20)
            (10, 14), (14, 16), (16, 18), (18, 20),
        ]
        
        # Add remaining connections for joints 21-25 if they exist
        if V > 21:
            edges.extend([
                (7, 21), (8, 22),  # Toes
                (12, 23), (12, 24), (12, 25)  # Head markers
            ])
        
        # Subset 0: Self-connections
        adjacency[0] = torch.eye(V)
        
        # Subset 1: Centripetal (child to parent)
        for parent, child in edges:
            if parent < V and child < V:
                adjacency[1, child, parent] = 1
        
        # Subset 2: Centrifugal (parent to child)
        for parent, child in edges:
            if parent < V and child < V:
                adjacency[2, parent, child] = 1
        
        # Normalize adjacency matrices
        for k in range(num_subsets):
            A = adjacency[k]
            D = A.sum(dim=1)
            D_inv_sqrt = torch.pow(D, -0.5)
            D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0
            adjacency[k] = D_inv_sqrt.unsqueeze(1) * A * D_inv_sqrt.unsqueeze(0)
        
        return adjacency
    
    def forward(self, x):
        """
        Args:
            x: (N, T, V, C) - batch, time, joints, coordinates
        Returns:
            dict with 'exercise' and 'correctness' logits
        """
        N, T, V, C = x.shape
        
        # Reshape to (N, C, T, V)
        x = x.permute(0, 3, 1, 2).contiguous()
        
        # Input normalization
        x = self.input_bn(x)
        
        # ST-GCN layers
        for stgcn in self.stgcn_layers:
            x = stgcn(x, self.base_adjacency)
        
        # Global pooling: (N, C, T, V) -> (N, C)
        features = self.global_pool(x).squeeze(-1).squeeze(-1)
        
        # Shared feature extraction
        features = self.feature_extractor(features)
        
        # Task-specific heads
        exercise_logits = self.exercise_classifier(features)
        correctness_logits = self.correctness_classifier(features).squeeze(-1)
        
        return {
            'exercise': exercise_logits,
            'correctness': correctness_logits,
            'features': features
        }


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss


class ImprovedSTGCNTrainer:
    """Training pipeline with fixes from BiLSTM"""
    
    def __init__(self, model, device='cpu', use_focal_loss=True, 
                 pos_weight=None):
        self.model = model.to(device)
        self.device = device
        
        # Loss functions
        if use_focal_loss:
            self.exercise_criterion = FocalLoss(alpha=1, gamma=2)
        else:
            self.exercise_criterion = nn.CrossEntropyLoss()
        
        # WEIGHTED correctness loss for class imbalance
        if pos_weight is not None:
            pos_weight_tensor = torch.tensor([pos_weight]).to(device)
            self.correctness_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        else:
            self.correctness_criterion = nn.BCEWithLogitsLoss()
        
        # Optimizer - same as BiLSTM
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=0.001,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # Scheduler - same as BiLSTM
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, 
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
        
        # History
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.train_ex_accuracies = []
        self.val_ex_accuracies = []
        self.learning_rates = []
    
    def reshape_input(self, x):
        """Reshape (N, T, F) to (N, T, V, C)"""
        N, T, F = x.shape
        num_joints = self.model.num_joints
        num_coords = F // num_joints
        x = x.view(N, T, num_joints, num_coords)
        return x
    
    def augment_data(self, x, exercise_ids):
        """
        Data augmentation for skeleton sequences
        - Random temporal cropping
        - Spatial jittering
        - Speed perturbation
        """
        if self.model.training and torch.rand(1).item() > 0.5:
            N, T, V, C = x.shape
            
            # Temporal cropping (keep 80-100% of frames)
            if T > 20:
                crop_ratio = 0.8 + torch.rand(1).item() * 0.2
                new_T = int(T * crop_ratio)
                start_idx = torch.randint(0, T - new_T + 1, (1,)).item()
                x = x[:, start_idx:start_idx+new_T, :, :]
            
            # Spatial jittering (small noise)
            noise = torch.randn_like(x) * 0.01
            x = x + noise
            
        return x
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        total_ex_loss = 0
        total_corr_loss = 0
        correct_correctness = 0
        correct_exercise = 0
        total_samples = 0
        
        for batch in train_loader:
            if 'rehab_joints' in batch and batch['rehab_joints'].numel() > 0:
                x = batch['rehab_joints'].to(self.device)
            elif 'mediapipe_joints' in batch and batch['mediapipe_joints'].numel() > 0:
                x = batch['mediapipe_joints'].to(self.device)
            else:
                continue
            
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            exercise_ids = batch['exercise_id'].to(self.device)
            correctness = batch['correctness'].float().to(self.device)
            
            # Reshape and augment
            x = self.reshape_input(x)
            x = self.augment_data(x, exercise_ids)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(x)
            
            # Calculate losses - SAME weighting as BiLSTM
            exercise_loss = self.exercise_criterion(outputs['exercise'], exercise_ids)
            correctness_loss = self.correctness_criterion(outputs['correctness'], correctness)
            loss = 0.2 * exercise_loss + 0.8 * correctness_loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            total_ex_loss += exercise_loss.item()
            total_corr_loss += correctness_loss.item()
            
            _, predicted_ex = torch.max(outputs['exercise'], 1)
            correct_exercise += (predicted_ex == exercise_ids).sum().item()
            
            predicted_corr = (torch.sigmoid(outputs['correctness']) > 0.5).float()
            correct_correctness += (predicted_corr == correctness).sum().item()
            
            total_samples += correctness.size(0)
        
        avg_loss = total_loss / len(train_loader)
        avg_ex_loss = total_ex_loss / len(train_loader)
        avg_corr_loss = total_corr_loss / len(train_loader)
        exercise_accuracy = correct_exercise / total_samples
        correctness_accuracy = correct_correctness / total_samples
        
        return avg_loss, correctness_accuracy, exercise_accuracy, avg_ex_loss, avg_corr_loss
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        total_ex_loss = 0
        total_corr_loss = 0
        correct_correctness = 0
        correct_exercise = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if 'rehab_joints' in batch and batch['rehab_joints'].numel() > 0:
                    x = batch['rehab_joints'].to(self.device)
                elif 'mediapipe_joints' in batch and batch['mediapipe_joints'].numel() > 0:
                    x = batch['mediapipe_joints'].to(self.device)
                else:
                    continue
                
                x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                exercise_ids = batch['exercise_id'].to(self.device)
                correctness = batch['correctness'].float().to(self.device)
                
                x = self.reshape_input(x)
                outputs = self.model(x)
                
                exercise_loss = self.exercise_criterion(outputs['exercise'], exercise_ids)
                correctness_loss = self.correctness_criterion(outputs['correctness'], correctness)
                loss = 0.2 * exercise_loss + 0.8 * correctness_loss
                
                total_loss += loss.item()
                total_ex_loss += exercise_loss.item()
                total_corr_loss += correctness_loss.item()
                
                _, predicted_ex = torch.max(outputs['exercise'], 1)
                correct_exercise += (predicted_ex == exercise_ids).sum().item()
                
                predicted_corr = (torch.sigmoid(outputs['correctness']) > 0.5).float()
                correct_correctness += (predicted_corr == correctness).sum().item()
                
                total_samples += correctness.size(0)
        
        avg_loss = total_loss / len(val_loader)
        avg_ex_loss = total_ex_loss / len(val_loader)
        avg_corr_loss = total_corr_loss / len(val_loader)
        exercise_accuracy = correct_exercise / total_samples
        correctness_accuracy = correct_correctness / total_samples
        
        return avg_loss, correctness_accuracy, exercise_accuracy, avg_ex_loss, avg_corr_loss
    
    def train(self, train_loader, val_loader, epochs=100, patience=20):
        best_val_corr_acc = 0.0
        patience_counter = 0
        
        print("Starting improved ST-GCN training...")
        print(f"Initial LR: {self.optimizer.param_groups[0]['lr']:.6f}")
        
        for epoch in range(epochs):
            train_loss, train_acc, train_ex_acc, train_ex_loss, train_corr_loss = self.train_epoch(train_loader)
            val_loss, val_acc, val_ex_acc, val_ex_loss, val_corr_loss = self.validate(val_loader)
            
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            
            # Save metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            self.train_ex_accuracies.append(train_ex_acc)
            self.val_ex_accuracies.append(val_ex_acc)
            
            print(f"Epoch {epoch+1:3d}/{epochs} | LR: {current_lr:.6f} | "
                  f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
                  f"Corr: {train_acc:.3f}/{val_acc:.3f} | "
                  f"Ex: {train_ex_acc:.3f}/{val_ex_acc:.3f} | "
                  f"ExL: {train_ex_loss:.3f}/{val_ex_loss:.3f} | "
                  f"CorrL: {train_corr_loss:.3f}/{val_corr_loss:.3f}")
            
            # Early stopping based on correctness accuracy
            if val_acc > best_val_corr_acc:
                best_val_corr_acc = val_acc
                patience_counter = 0
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'val_ex_acc': val_ex_acc
                }, os.path.join(MODEL_DIR, 'best_improved_stgcn.pth'))
                print(f"  ✓ Best model saved! Correctness: {val_acc:.3f}")
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        checkpoint = torch.load(os.path.join(MODEL_DIR, 'best_improved_stgcn.pth'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\nBest model: Val Loss {checkpoint['val_loss']:.4f}, "
              f"Correctness Acc {checkpoint['val_acc']:.3f}, "
              f"Exercise Acc {checkpoint['val_ex_acc']:.3f}")
    
    def plot_training_history(self):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        ax1.plot(self.train_losses, label='Train Loss', alpha=0.8, linewidth=2)
        ax1.plot(self.val_losses, label='Val Loss', alpha=0.8, linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=11)
        ax1.set_ylabel('Loss', fontsize=11)
        ax1.set_title('ST-GCN Training Loss', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Smoothed correctness accuracy
        window = 5
        if len(self.train_accuracies) > window:
            train_smooth = np.convolve(self.train_accuracies, np.ones(window)/window, mode='valid')
            val_smooth = np.convolve(self.val_accuracies, np.ones(window)/window, mode='valid')
            ax2.plot(range(window-1, len(self.train_accuracies)), train_smooth, 
                    label='Train Corr (smooth)', alpha=0.8, linewidth=2)
            ax2.plot(range(window-1, len(self.val_accuracies)), val_smooth, 
                    label='Val Corr (smooth)', alpha=0.8, linewidth=2)
        ax2.plot(self.train_accuracies, label='Train Corr', alpha=0.3, linewidth=1)
        ax2.plot(self.val_accuracies, label='Val Corr', alpha=0.3, linewidth=1)
        ax2.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random')
        ax2.set_xlabel('Epoch', fontsize=11)
        ax2.set_ylabel('Correctness Accuracy', fontsize=11)
        ax2.set_title('Correctness Detection', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3.plot(self.train_ex_accuracies, label='Train Ex', alpha=0.8, linewidth=2)
        ax3.plot(self.val_ex_accuracies, label='Val Ex', alpha=0.8, linewidth=2)
        ax3.set_xlabel('Epoch', fontsize=11)
        ax3.set_ylabel('Exercise Accuracy', fontsize=11)
        ax3.set_title('Exercise Classification', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        ax4.plot(self.learning_rates, linewidth=2, color='purple')
        ax4.set_xlabel('Epoch', fontsize=11)
        ax4.set_ylabel('Learning Rate', fontsize=11)
        ax4.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(GRAPH_DIR, 'improved_stgcn_history.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Training history saved to {GRAPH_DIR}/improved_stgcn_history.png")
    
    def evaluate_model(self, test_loader):
        """Detailed evaluation"""
        self.model.eval()
        
        all_exercise_preds = []
        all_exercise_true = []
        all_correctness_preds = []
        all_correctness_true = []
        all_correctness_probs = []
        
        with torch.no_grad():
            for batch in test_loader:
                if 'rehab_joints' in batch and batch['rehab_joints'].numel() > 0:
                    x = batch['rehab_joints'].to(self.device)
                elif 'mediapipe_joints' in batch and batch['mediapipe_joints'].numel() > 0:
                    x = batch['mediapipe_joints'].to(self.device)
                else:
                    continue
                
                x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                exercise_ids = batch['exercise_id']
                correctness = batch['correctness']
                
                x = self.reshape_input(x)
                outputs = self.model(x)
                
                _, predicted_ex = torch.max(outputs['exercise'], 1)
                correctness_probs = torch.sigmoid(outputs['correctness'])
                predicted_corr = (correctness_probs > 0.5).float()
                
                all_exercise_preds.extend(predicted_ex.cpu().numpy())
                all_exercise_true.extend(exercise_ids.numpy())
                all_correctness_preds.extend(predicted_corr.cpu().numpy())
                all_correctness_true.extend(correctness.numpy())
                all_correctness_probs.extend(correctness_probs.cpu().numpy())
        
        exercise_acc = accuracy_score(all_exercise_true, all_exercise_preds)
        correctness_acc = accuracy_score(all_correctness_true, all_correctness_preds)
        
        print(f"\n{'='*60}")
        print(f"IMPROVED ST-GCN TEST RESULTS")
        print(f"{'='*60}")
        print(f"Exercise Accuracy: {exercise_acc:.3f}")
        print(f"Correctness Accuracy: {correctness_acc:.3f}")
        print(f"Avg Confidence: {np.mean(all_correctness_probs):.3f}")
        
        exercise_names = ['Arm Abd.', 'Arm VW', 'Push-ups', 'Leg Abd.', 'Leg Lunge', 'Squats']
        print(f"\n{'-'*60}")
        print("Exercise Classification:")
        print(f"{'-'*60}")
        print(classification_report(all_exercise_true, all_exercise_preds, 
                                  target_names=exercise_names, digits=3))
        
        print(f"\n{'-'*60}")
        print("Correctness Detection:")
        print(f"{'-'*60}")
        print(classification_report(all_correctness_true, all_correctness_preds,
                                  target_names=['Incorrect', 'Correct'], digits=3))
        
        # Confusion matrices
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        cm = confusion_matrix(all_exercise_true, all_exercise_preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=exercise_names, yticklabels=exercise_names, ax=ax1)
        ax1.set_title('Exercise Classification', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('True')
        
        cm_corr = confusion_matrix(all_correctness_true, all_correctness_preds)
        sns.heatmap(cm_corr, annot=True, fmt='d', cmap='Greens',
                    xticklabels=['Incorrect', 'Correct'], 
                    yticklabels=['Incorrect', 'Correct'], ax=ax2)
        ax2.set_title('Correctness Detection', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('True')
        
        plt.tight_layout()
        plt.savefig(os.path.join(GRAPH_DIR, 'stgcn_confusion_matrices.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrices saved to {GRAPH_DIR}/stgcn_confusion_matrices.png")
        
        return exercise_acc, correctness_acc


def calculate_class_weights(train_loader):
    """Calculate class weights for imbalanced data"""
    all_correctness = []
    for batch in train_loader:
        all_correctness.extend(batch['correctness'].numpy())
    
    correct_count = sum(all_correctness)
    incorrect_count = len(all_correctness) - correct_count
    
    # pos_weight = num_negative / num_positive
    if correct_count > 0:
        pos_weight = incorrect_count / correct_count
    else:
        pos_weight = 1.0
    
    return pos_weight


def train_improved_stgcn():
    """Main training function with all improvements"""
    # Create dataloaders
    train_loader, val_loader, test_loader, dataset = create_dataloaders(
        batch_size=8,
        train_split=0.7,
        val_split=0.15
    )
    
    # Dataset statistics
    print("Dataset Statistics:")
    stats = dataset.get_exercise_stats()
    for ex, stat in stats.items():
        print(f"  {ex}: {stat}")
    
    # Calculate class imbalance
    pos_weight = calculate_class_weights(train_loader)
    print(f"\nClass imbalance - pos_weight: {pos_weight:.3f}")
    
    # Determine input dimensions
    for batch in train_loader:
        if 'rehab_joints' in batch and batch['rehab_joints'].numel() > 0:
            x = batch['rehab_joints']
            break
        elif 'mediapipe_joints' in batch and batch['mediapipe_joints'].numel() > 0:
            x = batch['mediapipe_joints']
            break
    
    N, T, F = x.shape
    num_joints = 26
    num_coords = F // num_joints
    
    print(f"\nInput configuration:")
    print(f"  Sequence shape: {x.shape}")
    print(f"  Number of joints: {num_joints}")
    print(f"  Coordinates per joint: {num_coords}")
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    model = ImprovedSTGCN(
        num_joints=num_joints,
        num_coords=num_coords,
        num_exercises=6,
        num_subsets=3,
        dropout=0.3,
        adaptive_graph=False  # Keep it simple, use fixed skeleton
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer with class weighting
    trainer = ImprovedSTGCNTrainer(
        model, 
        device, 
        use_focal_loss=True,
        pos_weight=pos_weight
    )
    
    # Train
    print("STARTING TRAINING")
    trainer.train(train_loader, val_loader, epochs=100, patience=20)
    
    # Evaluate
    print("FINAL EVALUATION")
    exercise_acc, correctness_acc = trainer.evaluate_model(test_loader)
    
    # Plot results
    trainer.plot_training_history()
    
    print("TRAINING COMPLETE")
    print(f"Final Exercise Accuracy: {exercise_acc:.3f}")
    print(f"Final Correctness Accuracy: {correctness_acc:.3f}")
    
    return trainer, test_loader, exercise_acc, correctness_acc


if __name__ == "__main__":
    trainer, test_loader, ex_acc, corr_acc = train_improved_stgcn()