# scripts/improved_exercise_model_v2.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from dataset_loader import create_dataloaders
import torch.nn.functional as F
import os

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
GRAPH_DIR = os.path.join(os.path.dirname(__file__), "graphs")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)

class ImprovedExerciseModel(nn.Module):
    
    def __init__(self, input_dim=52, hidden_dim=128, num_layers=2, 
                 num_exercises=6, dropout=0.5, use_attention=True):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_exercises = num_exercises
        self.use_attention = use_attention
        
        # Input normalization - use LayerNorm instead of BatchNorm for sequences
        self.input_norm = nn.LayerNorm(input_dim)
        
        # Bidirectional LSTM with reduced complexity
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Temporal attention for sequence modeling
        if use_attention:
            self.temporal_attention = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )
        
        # Feature extraction - shared features
        feature_dim = hidden_dim * 2  # bidirectional
        self.feature_extractor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # SEPARATE branches for each task with DIFFERENT architectures
        # Exercise classifier - simpler, already working well
        self.exercise_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, num_exercises)
        )
        
        self.correctness_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
            )
        
    def forward(self, x, lengths=None):
        batch_size, seq_len, _ = x.shape
        device = x.device  # Save device before packing
        
        # Input normalization
        x = self.input_norm(x)
        
        # Pack sequence if lengths provided
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Unpack if necessary
        if lengths is not None:
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        
        # Temporal attention-based pooling
        if self.use_attention:
            # Calculate attention weights
            attn_weights = self.temporal_attention(lstm_out)  # (batch, seq, 1)
            attn_weights = F.softmax(attn_weights, dim=1)
            
            # Apply attention mask if lengths provided
            if lengths is not None:
                mask = torch.arange(seq_len, device=device)[None, :] < lengths[:, None]
                attn_weights = attn_weights.masked_fill(~mask.unsqueeze(-1), 0)
                attn_weights = attn_weights / attn_weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
            
            # Weighted sum
            features = (lstm_out * attn_weights).sum(dim=1)  # (batch, hidden*2)
        else:
            # Use final hidden state
            features = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        
        # Shared feature extraction
        features = self.feature_extractor(features)
        
        # Task-specific predictions
        exercise_logits = self.exercise_classifier(features)
        correctness_logits = self.correctness_classifier(features).squeeze(-1)
        
        return {
            'exercise': exercise_logits,
            'correctness': correctness_logits,  # Raw logits for BCEWithLogitsLoss
            'features': features  # For analysis
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
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class ImprovedTrainer:
    """Improved training pipeline with better techniques"""
    
    def __init__(self, model, device='cpu', use_focal_loss=True):
        self.model = model.to(device)
        self.device = device
        
        # Loss functions - FIXED: Use BCEWithLogitsLoss
        if use_focal_loss:
            self.exercise_criterion = FocalLoss(alpha=1, gamma=2)
        else:
            self.exercise_criterion = nn.CrossEntropyLoss()
        
        # CRITICAL FIX: Use BCEWithLogitsLoss instead of BCELoss
        self.correctness_criterion = nn.BCEWithLogitsLoss()
        
        # Optimizer with HIGHER initial learning rate
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=0.001,  # INCREASED from 0.0001
            weight_decay=0.01,  # Reduced weight decay
            betas=(0.9, 0.999)
        )
        
        # Better learning rate scheduler - CosineAnnealingWarmRestarts
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, 
            T_0=10,  # Restart every 10 epochs
            T_mult=2,  # Double the period after each restart
            eta_min=1e-6
        )
        
        # Track learning rates
        self.learning_rates = []
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.train_ex_accuracies = []
        self.val_ex_accuracies = []
        
    def get_sequence_lengths(self, sequences):
        """Get actual sequence lengths (non-padded)"""
        # Find last non-zero frame for each sequence
        non_zero = (sequences.abs().sum(dim=-1) > 1e-6).long()
        lengths = non_zero.sum(dim=1)
        lengths = lengths.clamp(min=1)  # Ensure at least length 1
        return lengths
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        total_ex_loss = 0
        total_corr_loss = 0
        correct_correctness = 0
        correct_exercise = 0
        total_samples = 0
        
        for batch in train_loader:
            # Get data
            if 'rehab_joints' in batch and batch['rehab_joints'].numel() > 0:
                x = batch['rehab_joints'].to(self.device)
            elif 'mediapipe_joints' in batch and batch['mediapipe_joints'].numel() > 0:
                x = batch['mediapipe_joints'].to(self.device)
            else:
                continue
            
            # Handle NaN values
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            
            exercise_ids = batch['exercise_id'].to(self.device)
            correctness = batch['correctness'].float().to(self.device)
            
            # Get sequence lengths
            lengths = self.get_sequence_lengths(x)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(x, lengths=lengths)
            
            # Calculate losses
            exercise_loss = self.exercise_criterion(outputs['exercise'], exercise_ids)
            correctness_loss = self.correctness_criterion(outputs['correctness'], correctness)
            
            # DYNAMIC loss weighting - give more weight to harder task
            # Start with equal weights, gradually increase correctness weight
            loss = 0.2 * exercise_loss + 0.8 * correctness_loss
            
            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            total_ex_loss += exercise_loss.item()
            total_corr_loss += correctness_loss.item()
            
            # Exercise accuracy
            _, predicted_ex = torch.max(outputs['exercise'], 1)
            correct_exercise += (predicted_ex == exercise_ids).sum().item()
            
            # Correctness accuracy - use sigmoid for prediction
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
                
                lengths = self.get_sequence_lengths(x)
                
                outputs = self.model(x, lengths=lengths)
                
                exercise_loss = self.exercise_criterion(outputs['exercise'], exercise_ids)
                correctness_loss = self.correctness_criterion(outputs['correctness'], correctness)
                loss = 0.4 * exercise_loss + 0.6 * correctness_loss
                
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
        best_val_corr_acc = 0.0  # Track correctness accuracy instead of loss
        patience_counter = 0
        
        print("Starting improved training...")
        print(f"Initial LR: {self.optimizer.param_groups[0]['lr']:.6f}")
        
        for epoch in range(epochs):
            # Training
            train_loss, train_acc, train_ex_acc, train_ex_loss, train_corr_loss = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc, val_ex_acc, val_ex_loss, val_corr_loss = self.validate(val_loader)
            
            # Update scheduler AFTER each epoch
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
                }, os.path.join(MODEL_DIR, 'best_improved_exercise_model_2.pth'))
                print(f"  ✓ Best model saved! Correctness: {val_acc:.3f}")
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        checkpoint = torch.load('models/best_improved_exercise_model_2.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\nBest model: Val Loss {checkpoint['val_loss']:.4f}, "
              f"Correctness Acc {checkpoint['val_acc']:.3f}, "
              f"Exercise Acc {checkpoint['val_ex_acc']:.3f}")
    
    def plot_training_history(self):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss', alpha=0.8, linewidth=2)
        ax1.plot(self.val_losses, label='Val Loss', alpha=0.8, linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=11)
        ax1.set_ylabel('Loss', fontsize=11)
        ax1.set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Correctness accuracy with smoothing
        window = 5
        if len(self.train_accuracies) > window:
            train_smooth = np.convolve(self.train_accuracies, np.ones(window)/window, mode='valid')
            val_smooth = np.convolve(self.val_accuracies, np.ones(window)/window, mode='valid')
            ax2.plot(range(window-1, len(self.train_accuracies)), train_smooth, 
                    label='Train Corr Acc (smoothed)', alpha=0.8, linewidth=2)
            ax2.plot(range(window-1, len(self.val_accuracies)), val_smooth, 
                    label='Val Corr Acc (smoothed)', alpha=0.8, linewidth=2)
        ax2.plot(self.train_accuracies, label='Train Corr Acc', alpha=0.3, linewidth=1)
        ax2.plot(self.val_accuracies, label='Val Corr Acc', alpha=0.3, linewidth=1)
        ax2.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random Guess')
        ax2.set_xlabel('Epoch', fontsize=11)
        ax2.set_ylabel('Correctness Accuracy', fontsize=11)
        ax2.set_title('Correctness Detection Accuracy', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Exercise accuracy
        ax3.plot(self.train_ex_accuracies, label='Train Ex Acc', alpha=0.8, linewidth=2)
        ax3.plot(self.val_ex_accuracies, label='Val Ex Acc', alpha=0.8, linewidth=2)
        ax3.set_xlabel('Epoch', fontsize=11)
        ax3.set_ylabel('Exercise Accuracy', fontsize=11)
        ax3.set_title('Exercise Classification Accuracy', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Learning rate - NOW IT WILL CHANGE!
        ax4.plot(self.learning_rates, linewidth=2, color='purple')
        ax4.set_xlabel('Epoch', fontsize=11)
        ax4.set_ylabel('Learning Rate', fontsize=11)
        ax4.set_title('Learning Rate Schedule (Cosine Annealing)', fontsize=12, fontweight='bold')
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('improved_training_history_v2.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Training history saved as 'improved_training_history_v2.png'")
    
    def evaluate_model(self, test_loader):
        """Detailed model evaluation"""
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
                
                lengths = self.get_sequence_lengths(x)
                outputs = self.model(x, lengths=lengths)
                
                # Predictions
                _, predicted_ex = torch.max(outputs['exercise'], 1)
                correctness_probs = torch.sigmoid(outputs['correctness'])
                predicted_corr = (correctness_probs > 0.5).float()
                
                all_exercise_preds.extend(predicted_ex.cpu().numpy())
                all_exercise_true.extend(exercise_ids.numpy())
                all_correctness_preds.extend(predicted_corr.cpu().numpy())
                all_correctness_true.extend(correctness.numpy())
                all_correctness_probs.extend(correctness_probs.cpu().numpy())
        
        # Calculate metrics
        exercise_acc = accuracy_score(all_exercise_true, all_exercise_preds)
        correctness_acc = accuracy_score(all_correctness_true, all_correctness_preds)
        
        print(f"\n{'='*60}")
        print(f"FINAL TEST RESULTS")
        print(f"{'='*60}")
        print(f"Exercise Classification Accuracy: {exercise_acc:.3f}")
        print(f"Correctness Detection Accuracy: {correctness_acc:.3f}")
        print(f"Correctness Confidence (avg): {np.mean(all_correctness_probs):.3f}")
        
        # Exercise classification report
        exercise_names = ['Arm Abd.', 'Arm VW', 'Push-ups', 'Leg Abd.', 'Leg Lunge', 'Squats']
        print(f"\n{'-'*60}")
        print("Exercise Classification Report:")
        print(f"{'-'*60}")
        print(classification_report(all_exercise_true, all_exercise_preds, 
                                  target_names=exercise_names, digits=3))
        
        # Correctness report
        print(f"\n{'-'*60}")
        print("Correctness Detection Report:")
        print(f"{'-'*60}")
        print(classification_report(all_correctness_true, all_correctness_preds,
                                  target_names=['Incorrect', 'Correct'], digits=3))
        
        # Confusion matrix
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        cm = confusion_matrix(all_exercise_true, all_exercise_preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=exercise_names, yticklabels=exercise_names, ax=ax1)
        ax1.set_title('Exercise Classification Confusion Matrix', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Predicted', fontsize=11)
        ax1.set_ylabel('True', fontsize=11)
        
        cm_corr = confusion_matrix(all_correctness_true, all_correctness_preds)
        sns.heatmap(cm_corr, annot=True, fmt='d', cmap='Greens',
                    xticklabels=['Incorrect', 'Correct'], 
                    yticklabels=['Incorrect', 'Correct'], ax=ax2)
        ax2.set_title('Correctness Detection Confusion Matrix', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Predicted', fontsize=11)
        ax2.set_ylabel('True', fontsize=11)
        
        plt.tight_layout()
        plt.savefig('confusion_matrices_v3.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nConfusion matrices saved as 'confusion_matrices_v3.png'")
        
        return exercise_acc, correctness_acc

# Training script
def train_improved_model():
    # Create dataloaders
    train_loader, val_loader, test_loader, dataset = create_dataloaders(
        batch_size=8,  # Increase batch size slightly
        train_split=0.7,
        val_split=0.15
    )
    
    # Print data info
    print("Dataset Statistics:")
    stats = dataset.get_exercise_stats()
    for ex, stat in stats.items():
        print(f"  {ex}: {stat}")
    
    # Check class balance for correctness
    all_correctness = []
    for batch in train_loader:
        all_correctness.extend(batch['correctness'].numpy())
    correct_ratio = np.mean(all_correctness)
    print(f"\nCorrectness class balance: {correct_ratio:.2%} correct, {1-correct_ratio:.2%} incorrect")
    if abs(correct_ratio - 0.5) > 0.2:
        print("WARNING: Significant class imbalance detected!")
    
    # Determine input dimension
    for batch in train_loader:
        if 'rehab_joints' in batch and batch['rehab_joints'].numel() > 0:
            input_dim = batch['rehab_joints'].shape[-1]
            print(f"Using REHAB joints, input_dim: {input_dim}")
            break
        elif 'mediapipe_joints' in batch and batch['mediapipe_joints'].numel() > 0:
            input_dim = batch['mediapipe_joints'].shape[-1]
            print(f"Using MediaPipe joints, input_dim: {input_dim}")
            break
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = ImprovedExerciseModel(
        input_dim=input_dim,
        hidden_dim=128,  # Reduced from 256
        num_layers=2,    # Reduced from 3
        num_exercises=6,
        dropout=0.3,     # Reduced from 0.4
        use_attention=True
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = ImprovedTrainer(model, device, use_focal_loss=True)
    
    # Train
    trainer.train(train_loader, val_loader, epochs=100, patience=20)
    
    # Evaluate
    exercise_acc, correctness_acc = trainer.evaluate_model(test_loader)
    
    # Plot results
    trainer.plot_training_history()
    
    return trainer, test_loader, exercise_acc, correctness_acc

if __name__ == "__main__":
    trainer, test_loader, ex_acc, corr_acc = train_improved_model()
    print(f"FINAL RESULTS")
    print(f"Exercise Accuracy: {ex_acc:.3f}")
    print(f"Correctness Accuracy: {corr_acc:.3f}")