import numpy as np
import matplotlib.pyplot as plt
from scoring import SimplifiedReward
from collections import defaultdict
import seaborn as sns

def test_score_distribution():
    # Initialize reward model
    reward_model = SimplifiedReward()
    
    # Define test ranges
    response_times = np.concatenate([
        np.arange(20, 40, 5),  # 20s to 35s
        np.arange(40, 121, 5)  # 40s to 120s
    ])
    accuracy_rates = np.arange(0.50, 1.01, 0.05)  # 50% to 100% in 5% intervals
    
    # Store results for each group
    time_groups = defaultdict(list)    # Group by time, vary accuracy
    accuracy_groups = defaultdict(list) # Group by accuracy, vary time
    
    # Test hotkeys
    time_hotkeys = {t: f"time_{t}" for t in response_times}
    accuracy_hotkeys = {a: f"acc_{a}" for a in accuracy_rates}
    
    # Test each response time
    for time in response_times:
        # Test against all accuracy rates
        for accuracy in accuracy_rates:
            # Fill history with this time but varying accuracy
            hotkey = time_hotkeys[time]
            for _ in range(40):  # Fill the entire history
                predictions_match = np.random.random() < accuracy
                score = reward_model.calculate_score(time, predictions_match, hotkey)
            # Record final score
            time_groups[time].append(score)
    
    # Test each accuracy rate
    for accuracy in accuracy_rates:
        # Test against all response times
        for time in response_times:
            # Fill history with this accuracy but varying time
            hotkey = accuracy_hotkeys[accuracy]
            for _ in range(40):  # Fill the entire history
                predictions_match = np.random.random() < accuracy
                score = reward_model.calculate_score(time, predictions_match, hotkey)
            # Record final score
            accuracy_groups[accuracy].append(score)
    
    # Create visualizations
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Response Time Distribution
    plt.subplot(2, 1, 1)
    for time in response_times:
        scores = time_groups[time]
        plt.boxplot(scores, positions=[time], widths=3)
    plt.title('Score Distribution by Response Time (Varying Accuracy 50-100%)')
    plt.xlabel('Response Time (s)')
    plt.ylabel('Score')
    
    # Plot 2: Accuracy Rate Distribution
    plt.subplot(2, 1, 2)
    positions = np.arange(len(accuracy_rates))
    for i, accuracy in enumerate(accuracy_rates):
        scores = accuracy_groups[accuracy]
        plt.boxplot(scores, positions=[i], widths=0.5)
    plt.title('Score Distribution by Accuracy Rate (Varying Response Time 20-120s)')
    plt.xlabel('Accuracy Rate')
    plt.ylabel('Score')
    plt.xticks(positions, [f'{a:.0%}' for a in accuracy_rates], rotation=45)
    
    plt.tight_layout()
    plt.savefig('score_distributions.png')
    plt.close()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("\nBy Response Time (averaged across all accuracy rates):")
    for time in sorted(time_groups.keys()):
        scores = time_groups[time]
        print(f"Time {time}s: Mean={np.mean(scores):.3f}, Std={np.std(scores):.3f}, Min={np.min(scores):.3f}, Max={np.max(scores):.3f}")
    
    print("\nBy Accuracy Rate (averaged across all response times):")
    for accuracy in accuracy_rates:
        scores = accuracy_groups[accuracy]
        print(f"Accuracy {accuracy:.0%}: Mean={np.mean(scores):.3f}, Std={np.std(scores):.3f}, Min={np.min(scores):.3f}, Max={np.max(scores):.3f}")

if __name__ == "__main__":
    test_score_distribution()