# scripts/improved_exercise_model.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from dataset_loader import create_dataloaders
import torch.nn.functional as F

class ImprovedExerciseModel(nn.Module):
    """Improved multi-headed model with attention and better architecture"""
    
    def __init__(self, input_dim=52, hidden_dim=256, num_layers=3, 
                 num_exercises=6, dropout=0.4, use_attention=True):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_exercises = num_exercises
        self.use_attention = use_attention
        
        # Input normalization
        self.input_norm = nn.LayerNorm(input_dim)
        
        # Bidirectional LSTM for better temporal modeling
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim * 2,  # *2 for bidirectional
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
        
        # Feature extraction layers
        feature_dim = hidden_dim * 2  # bidirectional
        self.feature_extractor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Exercise classification head
        self.exercise_classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, num_exercises)
        )
        
        # Correctness classification head
        self.correctness_classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, mask=None):
        # x: (batch, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape
        
        # Input normalization
        x = self.input_norm(x)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)  # (batch, seq, hidden*2)
        
        # Apply attention if enabled
        if self.use_attention:
            # Self-attention over sequence
            attended_out, attention_weights = self.attention(lstm_out, lstm_out, lstm_out)
            # Global average pooling with attention
            if mask is not None:
                mask = mask.unsqueeze(-1).expand_as(attended_out)
                attended_out = attended_out * mask
                features = attended_out.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                features = attended_out.mean(dim=1)  # (batch, hidden*2)
        else:
            # Use final hidden state
            features = torch.cat([hidden[-2], hidden[-1]], dim=-1)  # Concat forward & backward
        
        # Feature extraction
        features = self.feature_extractor(features)
        
        # Predictions
        exercise_pred = self.exercise_classifier(features)
        correctness_pred = self.correctness_classifier(features)
        
        return {
            'exercise': exercise_pred,
            'correctness': correctness_pred.squeeze(-1)
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
        
        # Loss functions
        if use_focal_loss:
            self.exercise_criterion = FocalLoss(alpha=1, gamma=2)
        else:
            self.exercise_criterion = nn.CrossEntropyLoss()
        
        self.correctness_criterion = nn.BCELoss()
        
        # Optimizer with weight decay
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=0.0001,  # Lower learning rate
            weight_decay=0.02,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5, 
            verbose=True
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.train_ex_accuracies = []
        self.val_ex_accuracies = []
        
    def create_mask(self, sequences):
        """Create attention mask for padded sequences"""
        # Assume padded values are all zeros
        mask = (sequences.sum(dim=-1) != 0).float()  # (batch, seq_len)
        return mask
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        correct_correctness = 0
        correct_exercise = 0
        total_samples = 0
        
        for batch in train_loader:
            # Get the best available data
            if 'rehab_joints' in batch and batch['rehab_joints'].numel() > 0:
                x = batch['rehab_joints'].to(self.device)
            elif 'mediapipe_joints' in batch and batch['mediapipe_joints'].numel() > 0:
                x = batch['mediapipe_joints'].to(self.device)
            else:
                continue
            
            # Handle NaN values
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            
            exercise_ids = batch['exercise_id'].to(self.device)
            correctness = batch['correctness'].to(self.device)
            
            # Create attention mask
            mask = self.create_mask(x)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(x, mask=mask.to(self.device))
            
            # Calculate losses
            exercise_loss = self.exercise_criterion(outputs['exercise'], exercise_ids)
            correctness_loss = self.correctness_criterion(outputs['correctness'], correctness)
            
            # Weighted combined loss
            loss = 0.6 * exercise_loss + 0.4 * correctness_loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            
            # Exercise accuracy
            _, predicted_ex = torch.max(outputs['exercise'], 1)
            correct_exercise += (predicted_ex == exercise_ids).sum().item()
            
            # Correctness accuracy
            predicted_corr = (outputs['correctness'] > 0.5).float()
            correct_correctness += (predicted_corr == correctness).sum().item()
            
            total_samples += correctness.size(0)
        
        avg_loss = total_loss / len(train_loader)
        exercise_accuracy = correct_exercise / total_samples
        correctness_accuracy = correct_correctness / total_samples
        
        return avg_loss, correctness_accuracy, exercise_accuracy
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        correct_correctness = 0
        correct_exercise = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
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
                correctness = batch['correctness'].to(self.device)
                
                # Create mask
                mask = self.create_mask(x)
                
                # Forward pass
                outputs = self.model(x, mask=mask.to(self.device))
                
                # Calculate losses
                exercise_loss = self.exercise_criterion(outputs['exercise'], exercise_ids)
                correctness_loss = self.correctness_criterion(outputs['correctness'], correctness)
                loss = 0.6 * exercise_loss + 0.4 * correctness_loss
                
                # Statistics
                total_loss += loss.item()
                
                # Exercise accuracy
                _, predicted_ex = torch.max(outputs['exercise'], 1)
                correct_exercise += (predicted_ex == exercise_ids).sum().item()
                
                # Correctness accuracy
                predicted_corr = (outputs['correctness'] > 0.5).float()
                correct_correctness += (predicted_corr == correctness).sum().item()
                
                total_samples += correctness.size(0)
        
        avg_loss = total_loss / len(val_loader)
        exercise_accuracy = correct_exercise / total_samples
        correctness_accuracy = correct_correctness / total_samples
        
        return avg_loss, correctness_accuracy, exercise_accuracy
    
    def train(self, train_loader, val_loader, epochs=100, patience=15):
        best_val_loss = float('inf')
        patience_counter = 0
        
        print("Starting improved training...")
        
        for epoch in range(epochs):
            # Training
            train_loss, train_acc, train_ex_acc = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc, val_ex_acc = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Save metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            self.train_ex_accuracies.append(train_ex_acc)
            self.val_ex_accuracies.append(val_ex_acc)
            
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
                  f"Corr Acc: {train_acc:.3f}/{val_acc:.3f} | "
                  f"Ex Acc: {train_ex_acc:.3f}/{val_ex_acc:.3f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'val_ex_acc': val_ex_acc
                }, 'best_improved_exercise_model.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        checkpoint = torch.load('best_improved_exercise_model.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Best model: Val Loss {checkpoint['val_loss']:.4f}, "
              f"Correctness Acc {checkpoint['val_acc']:.3f}, "
              f"Exercise Acc {checkpoint['val_ex_acc']:.3f}")
    
    def plot_training_history(self):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss', alpha=0.8)
        ax1.plot(self.val_losses, label='Val Loss', alpha=0.8)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Correctness accuracy
        ax2.plot(self.train_accuracies, label='Train Corr Acc', alpha=0.8)
        ax2.plot(self.val_accuracies, label='Val Corr Acc', alpha=0.8)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Correctness Accuracy')
        ax2.set_title('Correctness Detection Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Exercise accuracy
        ax3.plot(self.train_ex_accuracies, label='Train Ex Acc', alpha=0.8)
        ax3.plot(self.val_ex_accuracies, label='Val Ex Acc', alpha=0.8)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Exercise Accuracy')
        ax3.set_title('Exercise Classification Accuracy')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Learning rate
        ax4.plot([group['lr'] for group in self.optimizer.param_groups] * len(self.train_losses))
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate')
        ax4.set_title('Learning Rate Schedule')
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('improved_training_history.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Improved training history saved as 'improved_training_history.png'")
    
    def evaluate_model(self, test_loader):
        """Detailed model evaluation"""
        self.model.eval()
        
        all_exercise_preds = []
        all_exercise_true = []
        all_correctness_preds = []
        all_correctness_true = []
        
        with torch.no_grad():
            for batch in test_loader:
                # Get data
                if 'rehab_joints' in batch and batch['rehab_joints'].numel() > 0:
                    x = batch['rehab_joints'].to(self.device)
                elif 'mediapipe_joints' in batch and batch['mediapipe_joints'].numel() > 0:
                    x = batch['mediapipe_joints'].to(self.device)
                else:
                    continue
                
                x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                exercise_ids = batch['exercise_id']
                correctness = batch['correctness']
                
                # Create mask
                mask = self.create_mask(x)
                
                # Predict
                outputs = self.model(x, mask=mask.to(self.device))
                
                # Collect predictions
                _, predicted_ex = torch.max(outputs['exercise'], 1)
                predicted_corr = (outputs['correctness'] > 0.5).float()
                
                all_exercise_preds.extend(predicted_ex.cpu().numpy())
                all_exercise_true.extend(exercise_ids.numpy())
                all_correctness_preds.extend(predicted_corr.cpu().numpy())
                all_correctness_true.extend(correctness.numpy())
        
        # Calculate metrics
        exercise_acc = accuracy_score(all_exercise_true, all_exercise_preds)
        correctness_acc = accuracy_score(all_correctness_true, all_correctness_preds)
        
        print(f"\nTest Results:")
        print(f"Exercise Classification Accuracy: {exercise_acc:.3f}")
        print(f"Correctness Detection Accuracy: {correctness_acc:.3f}")
        
        # Exercise classification report
        exercise_names = ['Arm Abd.', 'Arm VW', 'Push-ups', 'Leg Abd.', 'Leg Lunge', 'Squats']
        print(f"\nExercise Classification Report:")
        print(classification_report(all_exercise_true, all_exercise_preds, 
                                  target_names=exercise_names))
        
        # Confusion matrix
        cm = confusion_matrix(all_exercise_true, all_exercise_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=exercise_names, yticklabels=exercise_names)
        plt.title('Exercise Classification Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig('exercise_confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Confusion matrix saved as 'exercise_confusion_matrix.png'")
        
        return exercise_acc, correctness_acc

# Training script
def train_improved_model():
    # Create dataloaders with smaller batch size for better gradients
    train_loader, val_loader, test_loader, dataset = create_dataloaders(
        batch_size=4,  # Smaller batch size
        train_split=0.7,
        val_split=0.15
    )
    
    # Print data info
    print("Dataset Statistics:")
    stats = dataset.get_exercise_stats()
    for ex, stat in stats.items():
        print(f"  {ex}: {stat}")
    
    # Determine input dimension from first batch
    for batch in train_loader:
        if 'rehab_joints' in batch and batch['rehab_joints'].numel() > 0:
            input_dim = batch['rehab_joints'].shape[-1]
            print(f"Using REHAB joints, input_dim: {input_dim}")
            break
        elif 'mediapipe_joints' in batch and batch['mediapipe_joints'].numel() > 0:
            input_dim = batch['mediapipe_joints'].shape[-1]
            print(f"Using MediaPipe joints, input_dim: {input_dim}")
            break
    
    # Create improved model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = ImprovedExerciseModel(
        input_dim=input_dim,
        hidden_dim=256,
        num_layers=3,
        num_exercises=6,
        dropout=0.4,
        use_attention=True
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = ImprovedTrainer(model, device, use_focal_loss=True)
    
    # Train
    trainer.train(train_loader, val_loader, epochs=150, patience=25)
    
    # Evaluate
    exercise_acc, correctness_acc = trainer.evaluate_model(test_loader)
    
    # Plot results
    trainer.plot_training_history()
    
    return trainer, test_loader, exercise_acc, correctness_acc

if __name__ == "__main__":
    trainer, test_loader, ex_acc, corr_acc = train_improved_model()
    print(f"\nFinal Results: Exercise Acc: {ex_acc:.3f}, Correctness Acc: {corr_acc:.3f}")