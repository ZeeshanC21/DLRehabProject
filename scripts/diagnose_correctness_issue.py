# scripts/diagnose_correctness_issue.py
"""
CRITICAL: Run this BEFORE continuing model development
This will reveal why your models are stuck at 56% accuracy
"""

import numpy as np
from dataset_loader import create_dataloaders
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

def diagnose_correctness_problem():
    """Comprehensive diagnostic of correctness detection issue"""
    
    print("CORRECTNESS DETECTION - ROOT CAUSE ANALYSIS")
    print("="*70)
    
    # Load data
    train_loader, val_loader, test_loader, dataset = create_dataloaders(
        batch_size=1,
        train_split=0.7,
        val_split=0.15
    )
    
    # Collect all data
    all_data = {'train': [], 'val': [], 'test': []}
    loaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    
    for split_name, loader in loaders.items():
        for batch in loader:
            all_data[split_name].append({
                'exercise_id': batch['exercise_id'].item(),
                'correctness': batch['correctness'].item()
            })
    
    # ============================================================
    # TEST 1: Overall Class Distribution
    # ============================================================
    print("TEST 1: OVERALL CLASS DISTRIBUTION")
    print("="*70)
    
    for split_name in ['train', 'val', 'test']:
        data = all_data[split_name]
        correct_count = sum(1 for x in data if x['correctness'] == 1)
        total = len(data)
        correct_pct = 100 * correct_count / total if total > 0 else 0
        
        print(f"\n{split_name.upper()} SET:")
        print(f"  Total samples: {total}")
        print(f"  Correct: {correct_count} ({correct_pct:.1f}%)")
        print(f"  Incorrect: {total - correct_count} ({100-correct_pct:.1f}%)")
        
        # DIAGNOSIS
        if abs(correct_pct - 56) < 5:
            print(f"   WARNING: ~56% correct matches your model's 56% accuracy!")
            print(f"      → Model is likely just predicting majority class")
        
        if correct_pct > 70 or correct_pct < 30:
            print(f"   SEVERE IMBALANCE: {correct_pct:.1f}% correct")
            print(f"      → Need pos_weight = {(100-correct_pct)/correct_pct:.2f}")
    
    # ============================================================
    # TEST 2: Per-Exercise Breakdown
    # ============================================================
    print("TEST 2: PER-EXERCISE CORRECTNESS DISTRIBUTION")
    print("="*70)
    
    exercise_names = {
        0: 'Arm Abduction',
        1: 'Arm VW',
        2: 'Push-ups',
        3: 'Leg Abduction',
        4: 'Leg Lunge',
        5: 'Squats'
    }
    
    # Combine all splits for per-exercise analysis
    all_samples = []
    for split_data in all_data.values():
        all_samples.extend(split_data)
    
    exercise_stats = defaultdict(lambda: {'correct': 0, 'incorrect': 0})
    
    for sample in all_samples:
        ex_id = sample['exercise_id']
        if sample['correctness'] == 1:
            exercise_stats[ex_id]['correct'] += 1
        else:
            exercise_stats[ex_id]['incorrect'] += 1
    
    print("\nExercise-wise breakdown:")
    print("-" * 70)
    
    problem_exercises = []
    
    for ex_id in sorted(exercise_stats.keys()):
        stats = exercise_stats[ex_id]
        total = stats['correct'] + stats['incorrect']
        correct_pct = 100 * stats['correct'] / total if total > 0 else 0
        
        print(f"\n{exercise_names.get(ex_id, f'Exercise {ex_id}')}:")
        print(f"  Correct:   {stats['correct']:3d} samples ({correct_pct:.1f}%)")
        print(f"  Incorrect: {stats['incorrect']:3d} samples ({100-correct_pct:.1f}%)")
        print(f"  Total:     {total:3d} samples")
        
        # DIAGNOSIS
        if stats['incorrect'] < 10:
            print(f"  CRITICAL: Only {stats['incorrect']} incorrect samples!")
            print(f"     → Impossible to learn from <10 examples")
            problem_exercises.append(exercise_names.get(ex_id, f'Ex{ex_id}'))
        elif stats['incorrect'] < 20:
            print(f"   WARNING: Only {stats['incorrect']} incorrect samples")
            print(f"     → Very difficult to learn, needs more data")
            problem_exercises.append(exercise_names.get(ex_id, f'Ex{ex_id}'))
        
        if correct_pct > 80 or correct_pct < 20:
            print(f"   SEVERE IMBALANCE: {correct_pct:.1f}% vs {100-correct_pct:.1f}%")
    
    # ============================================================
    # TEST 3: Data Quality Checks
    # ============================================================
    print("TEST 3: DATA QUALITY CHECKS")
    print("="*70)
    
    # Check for any patterns
    train_samples = all_data['train']
    
    # Check if correctness correlates with exercise ID
    ex_corr_pattern = defaultdict(list)
    for sample in train_samples:
        ex_corr_pattern[sample['exercise_id']].append(sample['correctness'])
    
    print("\nChecking for suspicious patterns...")
    
    all_same = []
    for ex_id, correctness_list in ex_corr_pattern.items():
        unique_values = set(correctness_list)
        if len(unique_values) == 1:
            all_same.append((ex_id, list(unique_values)[0]))
    
    if all_same:
        print("  CRITICAL ISSUE: Some exercises have ONLY correct or ONLY incorrect!")
        for ex_id, value in all_same:
            label = "ALL CORRECT" if value == 1 else "ALL INCORRECT"
            print(f"     → {exercise_names.get(ex_id, f'Ex {ex_id}')}: {label}")
        print("     → Model cannot learn anything from these exercises")
    
    # ============================================================
    # TEST 4: Visualization
    # ============================================================
    print("TEST 4: GENERATING VISUALIZATIONS")
    print("="*70)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Overall distribution by split
    ax1 = axes[0, 0]
    splits = ['train', 'val', 'test']
    correct_counts = []
    incorrect_counts = []
    
    for split_name in splits:
        data = all_data[split_name]
        correct = sum(1 for x in data if x['correctness'] == 1)
        incorrect = len(data) - correct
        correct_counts.append(correct)
        incorrect_counts.append(incorrect)
    
    x = np.arange(len(splits))
    width = 0.35
    ax1.bar(x - width/2, correct_counts, width, label='Correct', color='green', alpha=0.7)
    ax1.bar(x + width/2, incorrect_counts, width, label='Incorrect', color='red', alpha=0.7)
    ax1.set_xlabel('Split')
    ax1.set_ylabel('Count')
    ax1.set_title('Correctness Distribution by Split')
    ax1.set_xticks(x)
    ax1.set_xticklabels(splits)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Per-exercise distribution
    ax2 = axes[0, 1]
    ex_ids = sorted(exercise_stats.keys())
    correct_by_ex = [exercise_stats[ex_id]['correct'] for ex_id in ex_ids]
    incorrect_by_ex = [exercise_stats[ex_id]['incorrect'] for ex_id in ex_ids]
    
    x = np.arange(len(ex_ids))
    ax2.bar(x - width/2, correct_by_ex, width, label='Correct', color='green', alpha=0.7)
    ax2.bar(x + width/2, incorrect_by_ex, width, label='Incorrect', color='red', alpha=0.7)
    ax2.set_xlabel('Exercise')
    ax2.set_ylabel('Count')
    ax2.set_title('Correctness Distribution by Exercise')
    ax2.set_xticks(x)
    ax2.set_xticklabels([exercise_names.get(ex_id, str(ex_id))[:10] for ex_id in ex_ids], 
                        rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Imbalance ratio per exercise
    ax3 = axes[1, 0]
    ratios = []
    for ex_id in ex_ids:
        correct = exercise_stats[ex_id]['correct']
        incorrect = exercise_stats[ex_id]['incorrect']
        ratio = correct / (incorrect + 1e-8)  # Avoid division by zero
        ratios.append(ratio)
    
    colors = ['red' if r > 3 or r < 0.33 else 'orange' if r > 2 or r < 0.5 else 'green' 
              for r in ratios]
    ax3.bar(x, ratios, color=colors, alpha=0.7)
    ax3.axhline(y=1.0, color='blue', linestyle='--', label='Balanced (1:1)')
    ax3.axhline(y=2.0, color='orange', linestyle='--', alpha=0.5, label='Warning (2:1)')
    ax3.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Exercise')
    ax3.set_ylabel('Correct/Incorrect Ratio')
    ax3.set_title('Class Balance per Exercise (Green=Good, Orange=Warning, Red=Critical)')
    ax3.set_xticks(x)
    ax3.set_xticklabels([exercise_names.get(ex_id, str(ex_id))[:10] for ex_id in ex_ids],
                        rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Sample count heatmap
    ax4 = axes[1, 1]
    heatmap_data = []
    for ex_id in ex_ids:
        heatmap_data.append([
            exercise_stats[ex_id]['correct'],
            exercise_stats[ex_id]['incorrect']
        ])
    
    sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlOrRd', 
                xticklabels=['Correct', 'Incorrect'],
                yticklabels=[exercise_names.get(ex_id, str(ex_id))[:15] for ex_id in ex_ids],
                ax=ax4, cbar_kws={'label': 'Sample Count'})
    ax4.set_title('Sample Count Heatmap\n(Red = Low count = Problem)')
    
    plt.tight_layout()
    plt.savefig('correctness_diagnosis.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved as 'correctness_diagnosis.png'")
    
    # ============================================================
    # FINAL DIAGNOSIS
    # ============================================================
    print("FINAL DIAGNOSIS & RECOMMENDATIONS")
    print("="*70)
    
    # Calculate overall statistics
    all_train = all_data['train']
    train_correct = sum(1 for x in all_train if x['correctness'] == 1)
    train_total = len(all_train)
    train_correct_pct = 100 * train_correct / train_total
    
    print("\nIdentified Issues:")
    issues = []
    
    if abs(train_correct_pct - 56) < 5:
        issues.append("CRITICAL: Overall ~56% correct = your model's 56% accuracy")
        print("  ISSUE #1: Class distribution matches model accuracy")
        print("     → Your model is just predicting the majority class!")
        print(f"     → It's getting {train_correct_pct:.1f}% by always predicting 'correct'")
    
    if problem_exercises:
        issues.append(f"CRITICAL: {len(problem_exercises)} exercises have <20 incorrect samples")
        print(f"  ISSUE #2: Insufficient data for {len(problem_exercises)} exercises")
        print(f"     → Affected: {', '.join(problem_exercises)}")
        print("     → Need at least 20-30 incorrect samples per exercise")
    
    min_incorrect = min(stats['incorrect'] for stats in exercise_stats.values())
    if min_incorrect < 5:
        issues.append("CRITICAL: Some exercises have <5 incorrect samples")
        print("  ISSUE #3: Some exercises have almost no incorrect samples")
        print("     → Impossible to learn from <5 examples")
    
    print("\n" + "-"*70)
    print("RECOMMENDATIONS:")
    print("-"*70)
    
    if abs(train_correct_pct - 56) < 5:
        print("\n1. USE WEIGHTED LOSS (CRITICAL):")
        pos_weight = (100 - train_correct_pct) / train_correct_pct
        print(f"   → Add to your trainer: pos_weight={pos_weight:.2f}")
        print("   → This forces model to care about minority class")
    
    if problem_exercises:
        print("\n2. COLLECT MORE INCORRECT SAMPLES:")
        print("   → Focus on exercises with <20 incorrect samples")
        print("   → Target: at least 30 incorrect per exercise")
        print("   → Or remove exercises with insufficient data")
    
    print("\n3. TRY TRAINING ON SINGLE EXERCISE:")
    print("   → Pick exercise with most balanced data")
    print("   → Train model on just that exercise")
    print("   → If it works (>70% acc), problem is data quantity")
    print("   → If it fails (~56% acc), problem is data quality/labels")
    
    print("\n4. VERIFY LABEL QUALITY:")
    print("   → Manually check 20 random 'incorrect' samples")
    print("   → Are they actually incorrect?")
    print("   → Is 'incorrect' well-defined?")
    
    if not issues:
        print("\n✓ Data looks reasonable - issue might be elsewhere")
        print("  → Check model capacity, learning rate, training duration")
    
    print("Run this diagnostic first, then fix data issues before model tuning!")
    
    return exercise_stats, all_data

if __name__ == "__main__":
    stats, data = diagnose_correctness_problem()