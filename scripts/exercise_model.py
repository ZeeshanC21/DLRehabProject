# scripts/improved_exercise_model_v2_balanced.py
"""
Version with class balancing strategies to handle imbalanced correctness labels
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from dataset_loader import create_dataloaders
import torch.nn.functional as F
import os
from collections import Counter

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
GRAPH_DIR = os.path.join(os.path.dirname(__file__), "graphs")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)

class ImprovedExerciseModel(nn.Module):
    """Model remains the same - architecture is fine"""
    
    def __init__(self, input_dim=52, hidden_dim=192, num_layers=2, 
                 num_exercises=6, dropout=0.2, use_attention=True):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_exercises = num_exercises
        self.use_attention = use_attention
        
        self.input_norm = nn.LayerNorm(input_dim)
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        if use_attention:
            self.temporal_attention = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )
        
        feature_dim = hidden_dim * 2
        self.feature_extractor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )
        
        self.exercise_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.25),
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
        device = x.device
        
        x = self.input_norm(x)
        
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        
        lstm_out, (hidden, cell) = self.lstm(x)
        
        if lengths is not None:
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        
        if self.use_attention:
            attn_weights = self.temporal_attention(lstm_out)
            attn_weights = F.softmax(attn_weights, dim=1)
            
            if lengths is not None:
                mask = torch.arange(seq_len, device=device)[None, :] < lengths[:, None]
                attn_weights = attn_weights.masked_fill(~mask.unsqueeze(-1), 0)
                attn_weights = attn_weights / (attn_weights.sum(dim=1, keepdim=True) + 1e-8)
            
            features = (lstm_out * attn_weights).sum(dim=1)
        else:
            features = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        
        features = self.feature_extractor(features)
        
        exercise_logits = self.exercise_classifier(features)
        correctness_logits = self.correctness_classifier(features).squeeze(-1)
        
        return {
            'exercise': exercise_logits,
            'correctness': correctness_logits,
            'features': features
        }

class BalancedTrainer:
    """Trainer with class balancing strategies"""
    
    def __init__(self, model, train_loader, device='cpu', 
                 exercise_weight=0.4, correctness_weight=0.6):
        self.model = model.to(device)
        self.device = device
        
        # SOLUTION 1: Calculate class weights for correctness

        print("COMPUTING CLASS WEIGHTS FOR BALANCED TRAINING")

        
        all_correctness = []
        all_exercises = []
        for batch in train_loader:
            all_correctness.extend(batch['correctness'].numpy())
            all_exercises.extend(batch['exercise_id'].numpy())
        
        all_correctness = np.array(all_correctness)
        all_exercises = np.array(all_exercises)
        
        # Compute per-exercise correctness imbalance
        exercise_correctness_weights = {}
        for ex_id in range(6):
            mask = all_exercises == ex_id
            ex_correctness = all_correctness[mask]
            if len(ex_correctness) > 0:
                correct_ratio = ex_correctness.mean()
                incorrect_ratio = 1 - correct_ratio
                # Weight for the minority class
                if correct_ratio > incorrect_ratio:
                    weight_correct = incorrect_ratio / correct_ratio
                    weight_incorrect = 1.0
                else:
                    weight_correct = 1.0
                    weight_incorrect = correct_ratio / incorrect_ratio
                exercise_correctness_weights[ex_id] = {
                    'correct': weight_correct,
                    'incorrect': weight_incorrect,
                    'ratio': max(correct_ratio, incorrect_ratio) / min(correct_ratio, incorrect_ratio)
                }
                print(f"Exercise {ex_id}: Correct={correct_ratio:.2%}, "
                      f"Weights=(correct:{weight_correct:.2f}, incorrect:{weight_incorrect:.2f}), "
                      f"Imbalance ratio: {exercise_correctness_weights[ex_id]['ratio']:.2f}")
        
        self.exercise_correctness_weights = exercise_correctness_weights
        
        # Overall class weights for correctness
        class_weights = compute_class_weight(
            'balanced', 
            classes=np.array([0, 1]), 
            y=all_correctness
        )
        self.correctness_pos_weight = torch.tensor(
            class_weights[1] / class_weights[0], 
            device=device
        )
        print(f"\nOverall correctness pos_weight: {self.correctness_pos_weight.item():.3f}")
        
        # Loss functions
        self.exercise_criterion = nn.CrossEntropyLoss()
        
        # CRITICAL FIX: Use pos_weight for class imbalance
        self.correctness_criterion = nn.BCEWithLogitsLoss(
            pos_weight=self.correctness_pos_weight
        )
        
        self.exercise_weight = exercise_weight
        self.correctness_weight = correctness_weight
        
        # SOLUTION 2: Lower learning rate for stability
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=0.0002,  # Even more conservative
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # SOLUTION 3: More patient scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=10,  # More patient
            verbose=True,
            min_lr=1e-6
        )
        
        # Tracking
        self.learning_rates = []
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.train_ex_accuracies = []
        self.val_ex_accuracies = []
        self.train_per_exercise_metrics = []
        self.val_per_exercise_metrics = []
        
    def get_sequence_lengths(self, sequences):
        non_zero = (sequences.abs().sum(dim=-1) > 1e-6).long()
        lengths = non_zero.sum(dim=1)
        lengths = lengths.clamp(min=1, max=sequences.shape[1])
        return lengths
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        total_ex_loss = 0
        total_corr_loss = 0
        correct_correctness = 0
        correct_exercise = 0
        total_samples = 0
        
        # Per-exercise tracking
        per_ex_correct = {i: 0 for i in range(6)}
        per_ex_total = {i: 0 for i in range(6)}
        per_ex_corr_correct = {i: 0 for i in range(6)}
        per_ex_corr_total = {i: 0 for i in range(6)}
        
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
            
            lengths = self.get_sequence_lengths(x)
            
            self.optimizer.zero_grad()
            outputs = self.model(x, lengths=lengths)
            
            # Calculate losses
            exercise_loss = self.exercise_criterion(outputs['exercise'], exercise_ids)
            correctness_loss = self.correctness_criterion(outputs['correctness'], correctness)
            
            loss = self.exercise_weight * exercise_loss + self.correctness_weight * correctness_loss
            
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
            
            # Per-exercise metrics
            for i in range(6):
                mask = exercise_ids == i
                if mask.sum() > 0:
                    per_ex_total[i] += mask.sum().item()
                    per_ex_correct[i] += (predicted_ex[mask] == exercise_ids[mask]).sum().item()
                    per_ex_corr_total[i] += mask.sum().item()
                    per_ex_corr_correct[i] += (predicted_corr[mask] == correctness[mask]).sum().item()
            
            total_samples += correctness.size(0)
        
        avg_loss = total_loss / len(train_loader)
        avg_ex_loss = total_ex_loss / len(train_loader)
        avg_corr_loss = total_corr_loss / len(train_loader)
        exercise_accuracy = correct_exercise / total_samples
        correctness_accuracy = correct_correctness / total_samples
        
        # Per-exercise accuracies
        per_ex_acc = {i: per_ex_correct[i]/per_ex_total[i] if per_ex_total[i] > 0 else 0 
                      for i in range(6)}
        per_ex_corr_acc = {i: per_ex_corr_correct[i]/per_ex_corr_total[i] if per_ex_corr_total[i] > 0 else 0 
                           for i in range(6)}
        
        return (avg_loss, correctness_accuracy, exercise_accuracy, 
                avg_ex_loss, avg_corr_loss, per_ex_acc, per_ex_corr_acc)
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        total_ex_loss = 0
        total_corr_loss = 0
        correct_correctness = 0
        correct_exercise = 0
        total_samples = 0
        
        # Per-exercise tracking
        per_ex_correct = {i: 0 for i in range(6)}
        per_ex_total = {i: 0 for i in range(6)}
        per_ex_corr_correct = {i: 0 for i in range(6)}
        per_ex_corr_total = {i: 0 for i in range(6)}
        
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
                loss = self.exercise_weight * exercise_loss + self.correctness_weight * correctness_loss
                
                total_loss += loss.item()
                total_ex_loss += exercise_loss.item()
                total_corr_loss += correctness_loss.item()
                
                _, predicted_ex = torch.max(outputs['exercise'], 1)
                correct_exercise += (predicted_ex == exercise_ids).sum().item()
                
                predicted_corr = (torch.sigmoid(outputs['correctness']) > 0.5).float()
                correct_correctness += (predicted_corr == correctness).sum().item()
                
                # Per-exercise metrics
                for i in range(6):
                    mask = exercise_ids == i
                    if mask.sum() > 0:
                        per_ex_total[i] += mask.sum().item()
                        per_ex_correct[i] += (predicted_ex[mask] == exercise_ids[mask]).sum().item()
                        per_ex_corr_total[i] += mask.sum().item()
                        per_ex_corr_correct[i] += (predicted_corr[mask] == correctness[mask]).sum().item()
                
                total_samples += correctness.size(0)
        
        avg_loss = total_loss / len(val_loader)
        avg_ex_loss = total_ex_loss / len(val_loader)
        avg_corr_loss = total_corr_loss / len(val_loader)
        exercise_accuracy = correct_exercise / total_samples
        correctness_accuracy = correct_correctness / total_samples
        
        per_ex_acc = {i: per_ex_correct[i]/per_ex_total[i] if per_ex_total[i] > 0 else 0 
                      for i in range(6)}
        per_ex_corr_acc = {i: per_ex_corr_correct[i]/per_ex_corr_total[i] if per_ex_corr_total[i] > 0 else 0 
                           for i in range(6)}
        
        return (avg_loss, correctness_accuracy, exercise_accuracy, 
                avg_ex_loss, avg_corr_loss, per_ex_acc, per_ex_corr_acc)
    
    def train(self, train_loader, val_loader, epochs=100, patience=30):
        best_val_combined = 0.0
        patience_counter = 0
        
        exercise_names = ['Arm Abd.', 'Arm VW', 'Push-ups', 'Leg Abd.', 'Leg Lunge', 'Squats']
        

        print("STARTING BALANCED TRAINING")

        print(f"Initial LR: {self.optimizer.param_groups[0]['lr']:.6f}")
        print(f"Loss weights - Exercise: {self.exercise_weight}, Correctness: {self.correctness_weight}")
        print(f"Patience: {patience} epochs")
        print("="*70 + "\n")
        
        for epoch in range(epochs):
            # Training
            (train_loss, train_acc, train_ex_acc, train_ex_loss, train_corr_loss, 
             train_per_ex, train_per_ex_corr) = self.train_epoch(train_loader)
            
            # Validation
            (val_loss, val_acc, val_ex_acc, val_ex_loss, val_corr_loss,
             val_per_ex, val_per_ex_corr) = self.validate(val_loader)
            
            # Combined metric
            combined_metric = 0.5 * val_ex_acc + 0.5 * val_acc
            self.scheduler.step(combined_metric)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            
            # Save metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            self.train_ex_accuracies.append(train_ex_acc)
            self.val_ex_accuracies.append(val_ex_acc)
            self.train_per_exercise_metrics.append((train_per_ex, train_per_ex_corr))
            self.val_per_exercise_metrics.append((val_per_ex, val_per_ex_corr))
            
            # Print progress
            print(f"Epoch {epoch+1:3d}/{epochs} | LR: {current_lr:.6f} | "
                  f"L: {train_loss:.3f}/{val_loss:.3f} | "
                  f"Corr: {train_acc:.3f}/{val_acc:.3f} | "
                  f"Ex: {train_ex_acc:.3f}/{val_ex_acc:.3f} | "
                  f"Comb: {combined_metric:.3f}")
            
            # SOLUTION 4: Print per-exercise correctness (watch for Squats!)
            if (epoch + 1) % 10 == 0:
                print("  Per-exercise correctness acc (val):")
                for i, name in enumerate(exercise_names):
                    acc = val_per_ex_corr[i]
                    symbol = "⚠️" if acc < 0.6 or acc > 0.9 else "✓"
                    print(f"    {symbol} {name}: {acc:.3f}")
            
            # Early stopping
            if combined_metric > best_val_combined:
                best_val_combined = combined_metric
                patience_counter = 0
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'val_ex_acc': val_ex_acc,
                    'combined_metric': combined_metric
                }, os.path.join(MODEL_DIR, 'best_balanced_model.pth'))
                print(f"  ✓ Best model saved! Combined: {combined_metric:.3f}")
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        # Load best model
        checkpoint = torch.load(os.path.join(MODEL_DIR, 'best_balanced_model.pth'))
        self.model.load_state_dict(checkpoint['model_state_dict'])

        print(f"Best model loaded:")
        print(f"  Combined: {checkpoint['combined_metric']:.3f}")
        print(f"  Correctness: {checkpoint['val_acc']:.3f}")
        print(f"  Exercise: {checkpoint['val_ex_acc']:.3f}")

    
    def plot_training_history(self):
        """Enhanced plotting with per-exercise metrics"""
        exercise_names = ['Arm Abd.', 'Arm VW', 'Push-ups', 'Leg Abd.', 'Leg Lunge', 'Squats']
        
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)
        
        # Row 1: Overall metrics
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(self.train_losses, label='Train', alpha=0.7)
        ax1.plot(self.val_losses, label='Val', alpha=0.9, linewidth=2)
        ax1.set_title('Total Loss', fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(self.train_accuracies, label='Train', alpha=0.5)
        ax2.plot(self.val_accuracies, label='Val', alpha=0.9, linewidth=2)
        ax2.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random')
        ax2.set_title('Correctness Accuracy', fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(self.train_ex_accuracies, label='Train', alpha=0.5)
        ax3.plot(self.val_ex_accuracies, label='Val', alpha=0.9, linewidth=2)
        ax3.set_title('Exercise Accuracy', fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy')
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        # Row 2 & 3: Per-exercise correctness accuracy
        for idx in range(6):
            ax = fig.add_subplot(gs[1 + idx//3, idx%3])
            
            train_accs = [metrics[1][idx] for metrics in self.train_per_exercise_metrics]
            val_accs = [metrics[1][idx] for metrics in self.val_per_exercise_metrics]
            
            ax.plot(train_accs, label='Train', alpha=0.5, linewidth=1)
            ax.plot(val_accs, label='Val', alpha=0.9, linewidth=2)
            ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
            ax.set_title(f'{exercise_names[idx]} - Correctness', fontsize=10, fontweight='bold')
            ax.set_xlabel('Epoch', fontsize=9)
            ax.set_ylabel('Accuracy', fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            ax.set_ylim([0, 1])
        
        # Row 4: Learning rate
        ax_lr = fig.add_subplot(gs[3, :])
        ax_lr.plot(self.learning_rates, linewidth=2, color='purple')
        ax_lr.set_title('Learning Rate Schedule', fontweight='bold')
        ax_lr.set_xlabel('Epoch')
        ax_lr.set_ylabel('Learning Rate')
        ax_lr.set_yscale('log')
        ax_lr.grid(alpha=0.3)
        
        plt.savefig(os.path.join(GRAPH_DIR, 'balanced_training_history.png'), 
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Training history saved to {os.path.join(GRAPH_DIR, 'balanced_training_history.png')}")
    
    def evaluate_model(self, test_loader):
        """Detailed evaluation with per-exercise breakdown"""
        self.model.eval()
        exercise_names = ['Arm Abd.', 'Arm VW', 'Push-ups', 'Leg Abd.', 'Leg Lunge', 'Squats']
        
        all_exercise_preds = []
        all_exercise_true = []
        all_correctness_preds = []
        all_correctness_true = []
        all_correctness_probs = []
        
        # Per-exercise storage
        per_ex_preds = {i: [] for i in range(6)}
        per_ex_true = {i: [] for i in range(6)}
        per_ex_probs = {i: [] for i in range(6)}
        
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
                
                _, predicted_ex = torch.max(outputs['exercise'], 1)
                correctness_probs = torch.sigmoid(outputs['correctness'])
                predicted_corr = (correctness_probs > 0.5).float()
                
                all_exercise_preds.extend(predicted_ex.cpu().numpy())
                all_exercise_true.extend(exercise_ids.numpy())
                all_correctness_preds.extend(predicted_corr.cpu().numpy())
                all_correctness_true.extend(correctness.numpy())
                all_correctness_probs.extend(correctness_probs.cpu().numpy())
                
                # Per-exercise collection
                for i in range(len(exercise_ids)):
                    ex_id = exercise_ids[i].item()
                    per_ex_preds[ex_id].append(predicted_corr[i].item())
                    per_ex_true[ex_id].append(correctness[i].item())
                    per_ex_probs[ex_id].append(correctness_probs[i].item())
        
        # Overall metrics
        exercise_acc = accuracy_score(all_exercise_true, all_exercise_preds)
        correctness_acc = accuracy_score(all_correctness_true, all_correctness_preds)
        

        print("FINAL TEST RESULTS")

        print(f"Exercise Classification Accuracy: {exercise_acc:.3f}")
        print(f"Correctness Detection Accuracy: {correctness_acc:.3f}")
        print(f"Combined Score: {0.5*exercise_acc + 0.5*correctness_acc:.3f}")
        
        # Per-exercise analysis

        print("PER-EXERCISE CORRECTNESS ANALYSIS")

        for i, name in enumerate(exercise_names):
            if len(per_ex_true[i]) > 0:
                ex_acc = accuracy_score(per_ex_true[i], per_ex_preds[i])
                ex_prob_mean = np.mean(per_ex_probs[i])
                ex_prob_std = np.std(per_ex_probs[i])
                true_ratio = np.mean(per_ex_true[i])
                
                status = "⚠️ PROBLEMATIC" if abs(ex_acc - 0.5) < 0.1 else "✓ Good"
                print(f"{status} {name}:")
                print(f"  Accuracy: {ex_acc:.3f}")
                print(f"  Avg Prob: {ex_prob_mean:.3f} ± {ex_prob_std:.3f}")
                print(f"  True Label Ratio (correct): {true_ratio:.3f}")
                print(f"  Samples: {len(per_ex_true[i])}")
        
        # Classification reports

        print("EXERCISE CLASSIFICATION REPORT")

        print(classification_report(all_exercise_true, all_exercise_preds, 
                                  target_names=exercise_names, digits=3))
        

        print("CORRECTNESS DETECTION REPORT")

        print(classification_report(all_correctness_true, all_correctness_preds,
                                  target_names=['Incorrect', 'Correct'], digits=3))
        
        # Confusion matrices
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        # Overall exercise confusion matrix
        cm = confusion_matrix(all_exercise_true, all_exercise_preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=exercise_names, yticklabels=exercise_names, ax=axes[0, 0])
        axes[0, 0].set_title('Exercise Classification', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('True')
        
        # Overall correctness confusion matrix
        cm_corr = confusion_matrix(all_correctness_true, all_correctness_preds)
        sns.heatmap(cm_corr, annot=True, fmt='d', cmap='Greens',
                    xticklabels=['Incorrect', 'Correct'], 
                    yticklabels=['Incorrect', 'Correct'], ax=axes[0, 1])
        axes[0, 1].set_title('Overall Correctness', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Predicted')
        axes[0, 1].set_ylabel('True')
        
        # Per-exercise correctness confusion matrices (first 6 exercises)
        for i in range(6):
            row = 0 if i < 2 else 1
            col = (i % 2) + 2 if i < 2 else i - 2
            ax = axes[row, col]
            
            if len(per_ex_true[i]) > 0:
                cm_ex = confusion_matrix(per_ex_true[i], per_ex_preds[i])
                sns.heatmap(cm_ex, annot=True, fmt='d', cmap='RdYlGn',
                           xticklabels=['Inc', 'Corr'], 
                           yticklabels=['Inc', 'Corr'], ax=ax)
                ax.set_title(f'{exercise_names[i]}', fontsize=10, fontweight='bold')
                ax.set_xlabel('Pred')
                ax.set_ylabel('True')
        
        plt.tight_layout()
        plt.savefig(os.path.join(GRAPH_DIR, 'confusion_matrices_balanced.png'), 
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nConfusion matrices saved to {os.path.join(GRAPH_DIR, 'confusion_matrices_balanced.png')}")
        
        return exercise_acc, correctness_acc

def train_balanced_model():
    """Main training function with balanced approach"""
    
    print("BALANCED TRAINING PIPELINE")
    
    # Create dataloaders with LARGER batch size
    train_loader, val_loader, test_loader, dataset = create_dataloaders(
        batch_size=16,  # Larger batch for stability
        train_split=0.7,
        val_split=0.15
    )
    
    # Dataset statistics
    print("\nDataset Statistics:")
    stats = dataset.get_exercise_stats()
    for ex, stat in stats.items():
        print(f"  {ex}: {stat}")
    
    # Check class balance
    all_correctness = []
    all_exercises = []
    for batch in train_loader:
        all_correctness.extend(batch['correctness'].numpy())
        all_exercises.extend(batch['exercise_id'].numpy())
    
    correct_ratio = np.mean(all_correctness)
    print(f"\nOverall correctness class balance:")
    print(f"  Correct: {correct_ratio:.2%}")
    print(f"  Incorrect: {1-correct_ratio:.2%}")
    
    # Per-exercise balance
    print("\nPer-exercise correctness balance:")
    exercise_names = ['Arm Abd.', 'Arm VW', 'Push-ups', 'Leg Abd.', 'Leg Lunge', 'Squats']
    all_correctness = np.array(all_correctness)
    all_exercises = np.array(all_exercises)
    
    for i, name in enumerate(exercise_names):
        mask = all_exercises == i
        ex_correct_ratio = all_correctness[mask].mean()
        imbalance_ratio = max(ex_correct_ratio, 1-ex_correct_ratio) / min(ex_correct_ratio, 1-ex_correct_ratio)
        status = "⚠️" if imbalance_ratio > 1.5 else "✓"
        print(f"  {status} {name}: {ex_correct_ratio:.2%} correct, ratio={imbalance_ratio:.2f}")
    
    # Determine input dimension
    for batch in train_loader:
        if 'rehab_joints' in batch and batch['rehab_joints'].numel() > 0:
            input_dim = batch['rehab_joints'].shape[-1]
            print(f"\nUsing REHAB joints, input_dim: {input_dim}")
            break
        elif 'mediapipe_joints' in batch and batch['mediapipe_joints'].numel() > 0:
            input_dim = batch['mediapipe_joints'].shape[-1]
            print(f"\nUsing MediaPipe joints, input_dim: {input_dim}")
            break
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = ImprovedExerciseModel(
        input_dim=input_dim,
        hidden_dim=192,
        num_layers=2,
        num_exercises=6,
        dropout=0.2,
        use_attention=True
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Create balanced trainer
    trainer = BalancedTrainer(
        model, 
        train_loader,  # Pass train_loader to compute class weights
        device, 
        exercise_weight=0.4,
        correctness_weight=0.6
    )
    
    # Train
    trainer.train(train_loader, val_loader, epochs=100, patience=30)
    
    # Evaluate
    exercise_acc, correctness_acc = trainer.evaluate_model(test_loader)
    
    # Plot results
    trainer.plot_training_history()
    
    return trainer, test_loader, exercise_acc, correctness_acc

if __name__ == "__main__":
    trainer, test_loader, ex_acc, corr_acc = train_balanced_model()
    print(f"FINAL SUMMARY")
    print(f"Exercise Accuracy: {ex_acc:.3f}")
    print(f"Correctness Accuracy: {corr_acc:.3f}")
    print(f"Combined Score: {0.5*ex_acc + 0.5*corr_acc:.3f}")