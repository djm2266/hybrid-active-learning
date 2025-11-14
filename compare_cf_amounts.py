#!/usr/bin/env python3
"""
Quick Comparison: Run with different CF shot amounts and compare F1 scores

Usage:
    python compare_cf_amounts.py

This will run 8 experiments with specific CF shot counts per round:
- 10 CF shots per round
- 15 CF shots per round
- 30 CF shots per round
- 50 CF shots per round
- 70 CF shots per round
- 90 CF shots per round
- 120 CF shots per round
- 0 CF shots (baseline)

And generate a comparison table with F1 scores and costs.
"""

import subprocess
import json
import pandas as pd
import os
from datetime import datetime

# CF shot counts to test (counterfactuals per round)
CF_SHOT_COUNTS = [10, 15, 30, 50, 70, 90, 120, 0]  # Added 0 as baseline


def update_config(cf_shots, human_shots=100):
    """Update config file with specific shot counts"""
    import yaml

    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Set budgets
    config['active_learning']['budget']['cf_budget_per_round'] = cf_shots
    config['active_learning']['budget']['human_budget_per_round'] = human_shots

    # Calculate cf_weight for routing logic
    total = cf_shots + human_shots
    if total > 0:
        cf_weight = cf_shots / total
    else:
        cf_weight = 0.0
    config['active_learning']['knob']['cf_weight'] = cf_weight

    with open('config/config.yaml', 'w') as f:
        yaml.dump(config, f)

    print(f"✓ Updated config: CF={cf_shots} shots/round, Human={human_shots} shots/round (cf_weight={cf_weight:.3f})")


def run_pipeline():
    """Run the pipeline"""
    result = subprocess.run(
        ['python', 'run_pipeline.py'],
        capture_output=True,
        text=True
    )
    return result.returncode == 0


def get_results():
    """Read results from output file"""
    results_file = 'data/output/al_final_results.json'

    if not os.path.exists(results_file):
        return None

    with open(results_file, 'r') as f:
        results = json.load(f)

    return results


def save_experiment_results(cf_shots, results, output_dir='data/output/comparison'):
    """Save results for this experiment"""
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{output_dir}/cf_{cf_shots}_shots_results.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

    return filename


def main():
    print("=" * 70)
    print("CF SHOT COUNT COMPARISON")
    print("=" * 70)
    print(f"\nThis will run {len(CF_SHOT_COUNTS)} experiments with different CF shot counts per round:")
    print(f"CF shots per round: {CF_SHOT_COUNTS}")
    print("\nEach experiment will take ~2-5 minutes depending on your dataset size.")
    print(f"Total estimated time: {len(CF_SHOT_COUNTS) * 3} minutes")
    print("\nPress Ctrl+C to cancel, or Enter to continue...")
    input()

    # Store all results
    all_results = []

    for i, cf_shots in enumerate(CF_SHOT_COUNTS, 1):
        print(f"\n{'=' * 70}")
        print(f"EXPERIMENT {i}/{len(CF_SHOT_COUNTS)}: CF Shots = {cf_shots} per round")
        print(f"{'=' * 70}\n")

        # Update config with specific shot counts
        human_shots = 100  # Keep human constant at 100
        update_config(cf_shots, human_shots)

        # Run pipeline
        print(f"Running pipeline with {cf_shots} CF shots per round...")
        success = run_pipeline()

        if not success:
            print(f"✗ Experiment failed for cf_shots={cf_shots}")
            continue

        # Get results
        results = get_results()
        if results:
            # Save this experiment
            save_experiment_results(cf_shots, results)

            # Extract key metrics
            summary = {
                'cf_shots_per_round': cf_shots,
                'human_shots_per_round': human_shots,
                'final_f1': results.get('final_f1', 0),
                'rounds': results.get('rounds', 0),
                'total_labeled': results.get('total_labeled', 0),
                'total_human': results.get('routing_summary', {}).get('total_human', 0),
                'total_cf': results.get('routing_summary', {}).get('total_cf', 0),
                'total_cost': results.get('total_cost', 0),
                'cf_acceptance': results.get('routing_summary', {}).get('avg_cf_acceptance', 0)
            }
            all_results.append(summary)

            print(f"\n✓ Completed: F1 = {summary['final_f1']:.4f}, Cost = ${summary['total_cost']:.2f}")
        else:
            print(f"✗ Could not read results for cf_shots={cf_shots}")

    if not all_results:
        print("\n✗ No experiments completed successfully!")
        return

    # Create comparison table
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)

    df = pd.DataFrame(all_results)

    # Sort by CF shots
    df = df.sort_values('cf_shots_per_round')

    # Calculate percentages
    df['cf_pct'] = (df['total_cf'] / df['total_labeled'] * 100).round(1)
    df['human_pct'] = (df['total_human'] / df['total_labeled'] * 100).round(1)

    # Display main comparison table
    print("\n📊 MAIN RESULTS TABLE")
    print("-" * 70)
    display_cols = ['cf_shots_per_round', 'final_f1', 'total_human', 'total_cf', 'total_cost', 'cf_acceptance']
    display_df = df[display_cols].copy()
    display_df.columns = ['CF Shots/Round', 'Final F1', 'Total Human', 'Total CF', 'Total Cost ($)', 'CF Accept Rate']
    print(display_df.to_string(index=False))
    print("-" * 70)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "data/output/comparison"
    os.makedirs(output_dir, exist_ok=True)

    output_file = f"{output_dir}/shot_comparison_{timestamp}.csv"
    df.to_csv(output_file, index=False)

    # Also create a simple summary table
    summary_file = f"{output_dir}/summary_table_{timestamp}.txt"
    with open(summary_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("CF SHOT COUNT VS F1 SCORE COMPARISON\n")
        f.write("=" * 70 + "\n\n")
        f.write(display_df.to_string(index=False))
        f.write("\n\n")

    print(f"\n✓ Results saved to:")
    print(f"   CSV: {output_file}")
    print(f"   Summary: {summary_file}")

    # Analysis
    print("\n" + "-" * 70)
    print("📈 ANALYSIS")
    print("-" * 70)

    best_f1_idx = df['final_f1'].idxmax()
    best_cost_idx = df['total_cost'].idxmin()

    best_f1 = df.iloc[best_f1_idx]
    best_cost = df.iloc[best_cost_idx]

    print(f"\n🏆 Best F1 Score: {best_f1['final_f1']:.4f}")
    print(f"   CF Shots per Round: {best_f1['cf_shots_per_round']}")
    print(f"   Total Human: {best_f1['total_human']}, Total CF: {best_f1['total_cf']}")
    print(f"   Total Cost: ${best_f1['total_cost']:.2f}")

    print(f"\n💰 Lowest Cost: ${best_cost['total_cost']:.2f}")
    print(f"   CF Shots per Round: {best_cost['cf_shots_per_round']}")
    print(f"   F1 Score: {best_cost['final_f1']:.4f}")
    print(f"   Total Human: {best_cost['total_human']}, Total CF: {best_cost['total_cf']}")

    # Compare to baseline (0 CF shots)
    if 0 in df['cf_shots_per_round'].values:
        baseline = df[df['cf_shots_per_round'] == 0].iloc[0]

        print(f"\n📊 Comparison to Baseline (0 CF shots):")
        print(f"   Baseline F1: {baseline['final_f1']:.4f}")
        print(f"   Baseline Cost: ${baseline['total_cost']:.2f}")

        if best_f1_idx != df[df['cf_shots_per_round'] == 0].index[0]:
            f1_improvement = ((best_f1['final_f1'] - baseline['final_f1']) / baseline['final_f1']) * 100
            print(f"   Best F1 improvement: {f1_improvement:+.1f}%")

        cost_savings = ((baseline['total_cost'] - best_cost['total_cost']) / baseline['total_cost']) * 100
        print(f"   Best cost savings: {cost_savings:.1f}%")

        # Show improvement for each configuration
        print(f"\n📈 F1 Improvement by CF Shot Count:")
        for _, row in df.iterrows():
            if row['cf_shots_per_round'] != 0:
                improvement = ((row['final_f1'] - baseline['final_f1']) / baseline['final_f1']) * 100
                cost_red = ((baseline['total_cost'] - row['total_cost']) / baseline['total_cost']) * 100
                print(
                    f"   {row['cf_shots_per_round']:3d} shots: F1 {improvement:+.1f}%, Cost {cost_red:.1f}% reduction")

    # Find sweet spot (best F1/cost ratio)
    df['f1_per_dollar'] = df['final_f1'] / (df['total_cost'] + 1)  # +1 to avoid division by zero
    best_ratio_idx = df['f1_per_dollar'].idxmax()
    best_ratio = df.iloc[best_ratio_idx]

    print(f"\n⚖️  Best F1/Cost Ratio:")
    print(f"   CF Shots per Round: {best_ratio['cf_shots_per_round']}")
    print(f"   F1: {best_ratio['final_f1']:.4f}, Cost: ${best_ratio['total_cost']:.2f}")
    print(f"   F1 per dollar: {best_ratio['f1_per_dollar']:.6f}")

    print("\n" + "=" * 70)
    print("✅ COMPLETE!")
    print("=" * 70 + "\n")

    return df


if __name__ == "__main__":
    main()