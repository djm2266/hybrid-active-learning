#!/usr/bin/env python3
"""
Main Pipeline Runner
Executes the full hybrid active learning pipeline
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from utils import load_config, ensure_directories
import pandas as pd
import os


def run_full_pipeline(config_path='config/config.yaml'):
    """Run the complete pipeline"""

    print("\n" + "="*70)
    print("HYBRID ACTIVE LEARNING PIPELINE")
    print("="*70)

    # Load config
    config = load_config(config_path)
    ensure_directories(config)

    # Import after path is set
    from src.active_learning_loop import HybridActiveLearning

    # Load data
    dirs = config['directories']
    dataset_config = config['dataset']

    train_path = f"{dirs['input_data']}/{dataset_config['train_file']}"
    test_path = f"{dirs['input_data']}/{dataset_config['test_file']}"

    if not os.path.exists(train_path):
        print(f"\nERROR: Training data not found: {train_path}")
        print("Please add your training data to data/input/")
        return 1

    print(f"\nLoading data...")
    train_df = pd.read_csv(train_path)
    print(f"  Training data: {len(train_df)} samples")

    # Split into initial labeled and unlabeled pool
    n_initial = config['active_learning'].get('initial_labeled_size', 100)
    initial_labeled = train_df.iloc[:n_initial]
    unlabeled_pool = train_df.iloc[n_initial:]

    print(f"  Initial labeled: {len(initial_labeled)}")
    print(f"  Unlabeled pool: {len(unlabeled_pool)}")

    # Load test data if available
    test_df = None
    if os.path.exists(test_path):
        test_df = pd.read_csv(test_path)
        print(f"  Test set: {len(test_df)}")

    # Run active learning
    print("\nInitializing Active Learning System...")
    al_system = HybridActiveLearning(config)

    print("\nStarting Active Learning Loop...")
    results = al_system.run(initial_labeled, unlabeled_pool, test_df)

    # Print final summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print(f"\nFinal Results:")
    print(f"  Rounds: {results['rounds']}")
    print(f"  Final F1: {results['final_f1']:.4f}")
    print(f"  Total labeled: {results['total_labeled']}")
    print(f"  Total cost: {results['total_cost']:.2f}")
    print(f"\nRouting Summary:")
    routing = results['routing_summary']
    print(f"  Human annotations: {routing.get('total_human', 0)} ({routing.get('human_pct', 0):.1f}%)")
    print(f"  CF generations: {routing.get('total_cf', 0)} ({routing.get('cf_pct', 0):.1f}%)")
    print(f"  Avg CF acceptance: {routing.get('avg_cf_acceptance', 0):.2f}")
    print("="*70 + "\n")

    return 0


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Run Hybrid Active Learning Pipeline'
    )
    parser.add_argument(
        '--config',
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--check-only',
        action='store_true',
        help='Only run setup checks'
    )

    args = parser.parse_args()

    if args.check_only:
        # Run checks
        import subprocess
        result = subprocess.run(['python', 'check_setup.py'])
        return result.returncode

    # Run pipeline
    try:
        return run_full_pipeline(args.config)
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user!")
        return 130
    except Exception as e:
        print(f"\n\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
