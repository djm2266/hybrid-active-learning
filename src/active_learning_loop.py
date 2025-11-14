#!/usr/bin/env python3
"""
Script 05: Hybrid Active Learning Loop
Implements Algorithm 1: Hybrid Active Learning with Knob For Counterfactuals and Human Oracle

This is the main orchestration script that:
1. Scores unlabeled samples
2. Computes dynamic thresholds
3. Routes samples (human/CF/defer)
4. Obtains labels from both sources
5. Updates model with weighted training
6. Adapts thresholds for next round
"""

import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import json
from datetime import datetime

from utils import load_config, ensure_directories, get_llm_provider
from active_learning.scorer import ActiveLearningScorer
from active_learning.router import ActiveLearningRouter
from active_learning.oracle import HumanOracle
from active_learning.trainer import WeightedModelTrainer
from counterfactual.generator import CounterfactualGenerator
from counterfactual.validator import CounterfactualValidator
from utils.embeddings import EmbeddingGenerator
from utils.metrics import compute_metrics


class HybridActiveLearning:
    """
    Main active learning orchestrator implementing Algorithm 1.
    """

    def __init__(self, config: dict):
        """
        Initialize active learning system.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.al_config = config['active_learning']

        # Initialize components
        self.scorer = ActiveLearningScorer(config)
        self.router = ActiveLearningRouter(config)
        self.trainer = WeightedModelTrainer(config)
        self.embedding_gen = EmbeddingGenerator(config)

        # LLM provider for CF generation
        self.llm_provider = get_llm_provider(config)
        self.cf_generator = CounterfactualGenerator(config, self.llm_provider)
        self.cf_validator = CounterfactualValidator(config)

        # Human oracle
        self.human_oracle = HumanOracle(config)

        # Data tracking
        self.labeled_data = []
        self.unlabeled_data = []
        self.round_history = []

        # Initialize model
        self.model = self.trainer.initialize_model()

    def run(
            self,
            initial_labeled: pd.DataFrame,
            unlabeled_pool: pd.DataFrame,
            test_data: pd.DataFrame = None
    ) -> Dict:
        """
        Run the full hybrid active learning loop.

        Args:
            initial_labeled: Initial labeled dataset
            unlabeled_pool: Unlabeled data pool
            test_data: Test set for evaluation

        Returns:
            Dictionary with final results and statistics
        """
        print("\n" + "=" * 70)
        print("HYBRID ACTIVE LEARNING WITH COUNTERFACTUALS AND HUMAN ORACLE")
        print("=" * 70)

        # Initialize
        self.labeled_data = initial_labeled.to_dict('records')
        self.unlabeled_data = unlabeled_pool.to_dict('records')

        # Train initial model
        print("\n=== Training Initial Model ===")
        self.model = self.trainer.train(
            self.labeled_data,
            sample_weights=[1.0] * len(self.labeled_data)
        )

        # Initial evaluation
        if test_data is not None:
            initial_metrics = self._evaluate_model(test_data)
            print(f"\nInitial F1: {initial_metrics['f1_macro']:.4f}")

        # Active learning rounds
        max_rounds = self.al_config['budget']['total_rounds']

        for round_num in range(1, max_rounds + 1):
            print(f"\n{'=' * 70}")
            print(f"ROUND {round_num}/{max_rounds}")
            print(f"{'=' * 70}")

            # Run one round
            round_result = self._run_round(round_num, test_data)
            self.round_history.append(round_result)

            # Check stopping criteria
            if self._should_stop(round_num):
                print(f"\n✓ Stopping criteria met after {round_num} rounds")
                break

        # Final results
        final_results = self._compile_results()
        self._save_results(final_results)

        print("\n" + "=" * 70)
        print("ACTIVE LEARNING COMPLETE")
        print("=" * 70)

        return final_results

    def _run_round(
            self,
            round_num: int,
            test_data: pd.DataFrame = None
    ) -> Dict:
        """
        Run a single active learning round following Algorithm 1.

        Args:
            round_num: Current round number
            test_data: Optional test set for evaluation

        Returns:
            Dictionary with round statistics
        """
        round_start = datetime.now()
        cf_generated = []  # <--- initialize so it always exists
        cf_acceptance_rate = 0.0

        # Step 1: Score each unlabeled item
        print(f"\nStep 1: Scoring {len(self.unlabeled_data)} unlabeled samples...")
        scores, embeddings = self._score_unlabeled_pool()

        # Step 2: Compute dynamic thresholds
        print("\nStep 2: Computing dynamic thresholds...")
        thresholds = self.scorer.compute_thresholds(scores)
        self._print_thresholds(thresholds)

        # Step 3: Route items for labeling
        print("\nStep 3: Routing samples...")
        human_samples, cf_samples, deferred = self.router.route_samples(
            self.unlabeled_data,
            scores,
            thresholds
        )

        # Enforce coverage quotas
        cluster_info = self.scorer.get_cluster_info()
        human_samples, cf_samples = self.router.enforce_coverage_quotas(
            human_samples, cf_samples, cluster_info
        )

        # Step 4: Obtain labels
        print("\nStep 4: Obtaining labels...")
        training_data = []

        # 4a: Human oracle annotations
        if human_samples:
            print(f"\n  Human oracle: {len(human_samples)} samples")
            human_labeled = self._obtain_human_labels(human_samples)
            training_data.extend(human_labeled)

        # 4b: Counterfactual generation
        if cf_samples:
            print(f"\n  Counterfactual generation: {len(cf_samples)} samples")
            cf_generated = self._generate_counterfactuals(cf_samples)
            training_data.extend(cf_generated)

        # Step 5: Update model
        print(f"\nStep 5: Updating model with {len(training_data)} new samples...")
        self.model = self._update_model(training_data)

        # Evaluate
        metrics = {}
        if test_data is not None:
            metrics = self._evaluate_model(test_data)
            print(f"\n  Global F1: {metrics['f1_macro']:.4f}")
            print(f"  Per-cluster F1: {metrics.get('per_cluster_f1', {})}")

        # Step 6: Adapt for next round
        print("\nStep 6: Adapting for next round...")
        self._update_data_pools(human_samples, cf_samples, training_data)
        self._adapt_thresholds()

        # Track CF acceptance rate for adaptive routing
        if cf_generated:
            cf_acceptance_rate = self._compute_cf_acceptance_rate(cf_generated)
        else:
            cf_acceptance_rate = 0.0
        self.router.update_cf_acceptance_rate(cf_acceptance_rate)

        # Compute costs
        costs = self._compute_round_costs(human_samples, cf_samples)

        round_time = (datetime.now() - round_start).total_seconds()

        # Round summary
        round_result = {
            'round': round_num,
            'human_samples': len(human_samples),
            'cf_samples': len(cf_samples),
            'cf_accepted': sum(1 for x in cf_generated if x.get('accepted')),
            'cf_acceptance_rate': cf_acceptance_rate,
            'deferred': len(deferred),
            'training_samples': len(training_data),
            'labeled_pool_size': len(self.labeled_data),
            'unlabeled_pool_size': len(self.unlabeled_data),
            'costs': costs,
            'metrics': metrics,
            'thresholds': thresholds,
            'time_seconds': round_time
        }

        # Save round checkpoint
        self._save_round_checkpoint(round_num, round_result)

        return round_result

    def _score_unlabeled_pool(self) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Score all unlabeled samples.

        Returns:
            Tuple of (scores dict, embeddings array)
        """
        # Generate embeddings
        texts = [s['example'] for s in self.unlabeled_data]
        embeddings = self.embedding_gen.encode(texts)

        # Get labeled embeddings for novelty
        if self.labeled_data:
            labeled_texts = [s['example'] for s in self.labeled_data]
            labeled_embeddings = self.embedding_gen.encode(labeled_texts)
        else:
            labeled_embeddings = None

        # Compute all scores
        scores = self.scorer.compute_all_scores(
            self.unlabeled_data,
            self.model,
            embeddings,
            labeled_embeddings
        )

        # Add cluster assignments to samples
        cluster_info = self.scorer.get_cluster_info()
        if 'assignments' in cluster_info:
            for idx, sample in enumerate(self.unlabeled_data):
                sample['cluster_id'] = int(cluster_info['assignments'][idx])

        return scores, embeddings

    def _obtain_human_labels(self, samples: List[dict]) -> List[dict]:
        """
        Obtain labels from human oracle.

        Args:
            samples: Samples to annotate

        Returns:
            List of labeled samples with weight=1.0
        """
        labeled = []

        for sample in samples:
            label = self.human_oracle.get_label(sample)

            if label is not None:
                labeled.append({
                    'id': sample['id'],
                    'example': sample['example'],
                    'Label': label,
                    'weight': self.al_config['sample_weights']['human_labeled'],
                    'source': 'human',
                    'original_id': sample['id']
                })

        print(f"    ✓ Obtained {len(labeled)} human annotations")
        return labeled

    def _generate_counterfactuals(self, samples: List[dict]) -> List[dict]:
        """
        Generate and validate counterfactuals.

        Args:
            samples: Samples for CF generation

        Returns:
            List of CF samples with weight~0.7 and acceptance flag
        """
        cf_samples = []
        accepted_count = 0

        # Optional "teacher forcing" mode: trust CF target labels more than current model
        cf_val_cfg = self.config.get('counterfactual', {}).get('validation', {})
        teacher_forcing = cf_val_cfg.get('teacher_forcing', False)

        # Stats for debugging/model-behavior metrics
        total_cf = 0
        pred_eq_target = 0
        pred_neq_target = 0
        rejection_stats: Dict[str, int] = {}

        for sample in samples:
            # Generate counterfactual
            cf_result = self.cf_generator.generate(sample, self.model)
            total_cf += 1

            if cf_result is None:
                cf_samples.append({
                    'original_id': sample['id'],
                    'accepted': False,
                    'rejection_reason': 'generation_failed'
                })
                rejection_stats['generation_failed'] = rejection_stats.get('generation_failed', 0) + 1
                continue

            # Validate counterfactual
            is_valid, validation_result = self.cf_validator.validate(
                original=sample,
                counterfactual=cf_result,
                model=self.model
            )

            pred = validation_result.get('pred_label')
            target = validation_result.get('cf_label')

            if pred == target:
                pred_eq_target += 1
            else:
                pred_neq_target += 1

            accepted = is_valid

            # --- Teacher forcing override ----------------------------------
            if teacher_forcing and not is_valid:
                # Relax dependence on the current classifier:
                sim_ok = validation_result.get('similarity_ok', True)
                leakage_free = validation_result.get('no_label_leakage', True)

                # We ignore label_flip and confidence in this override,
                # and accept as long as the text looks like a good CF.
                if sim_ok and leakage_free:
                    accepted = True
                    validation_result['forced_accept'] = True
                    validation_result['reason'] = validation_result.get('reason', 'forced_accept')
            # ----------------------------------------------------------------

            if accepted:
                cf_samples.append({
                    'id': f"cf_{sample['id']}",
                    'example': cf_result['example'],
                    # IMPORTANT: use the CF's intended target label
                    'Label': cf_result['Label'],
                    'weight': self.al_config['sample_weights']['cf_generated'],
                    'source': 'counterfactual',
                    'original_id': sample['id'],
                    'accepted': True,
                    'validation': validation_result
                })
                accepted_count += 1
            else:
                reason = validation_result.get('reason', 'unknown')
                cf_samples.append({
                    'original_id': sample['id'],
                    'accepted': False,
                    'rejection_reason': reason,
                    'validation': validation_result
                })
                rejection_stats[reason] = rejection_stats.get(reason, 0) + 1

        # Summary logging (per round)
        print(f"    ✓ Generated {accepted_count}/{len(samples)} valid counterfactuals")

        if total_cf > 0:
            print("    CF metrics (all generated CFs this round):")
            print(f"      total CFs:          {total_cf}")
            print(f"      model_pred == CF target_label: {pred_eq_target}")
            print(f"      model_pred != CF target_label: {pred_neq_target}")
            frac_match = pred_eq_target / total_cf
            print(f"      fraction match:     {frac_match:.3f}")

        if rejection_stats:
            print("    CF rejection breakdown:")
            for r, c in rejection_stats.items():
                print(f"      {r}: {c}")

        return cf_samples

    def _update_model(self, training_data: List[dict]) -> object:
        """
        Update model with weighted training data.

        Args:
            training_data: New training samples with weights

        Returns:
            Updated model
        """
        # Combine with existing labeled data
        all_training_data = self.labeled_data + [
            x for x in training_data if x.get('accepted', True)
        ]

        # Extract weights
        weights = [x.get('weight', 1.0) for x in all_training_data]

        # Train model
        model = self.trainer.train(all_training_data, weights)

        return model

    def _evaluate_model(self, test_data: pd.DataFrame) -> Dict:
        """
        Evaluate model on test set.

        Args:
            test_data: Test dataset

        Returns:
            Dictionary of metrics
        """
        test_samples = test_data.to_dict('records')
        texts = [s['example'] for s in test_samples]
        true_labels = [s['Label'] for s in test_samples]

        predictions = self.model.predict(texts)

        metrics = compute_metrics(true_labels, predictions)

        # Per-cluster metrics if available
        if 'cluster_id' in test_samples[0]:
            cluster_metrics = self._compute_per_cluster_metrics(
                test_samples, predictions
            )
            metrics['per_cluster_f1'] = cluster_metrics

        # Boundary health
        metrics['boundary_health'] = self._assess_boundary_health(test_samples)

        return metrics

    def _compute_per_cluster_metrics(
            self,
            samples: List[dict],
            predictions: List[str]
    ) -> Dict:
        """Compute F1 per cluster"""
        clusters = {}
        for sample, pred in zip(samples, predictions):
            cluster_id = sample.get('cluster_id')
            if cluster_id not in clusters:
                clusters[cluster_id] = {'true': [], 'pred': []}
            clusters[cluster_id]['true'].append(sample['label'])
            clusters[cluster_id]['pred'].append(pred)

        cluster_f1 = {}
        for cluster_id, data in clusters.items():
            metrics = compute_metrics(data['true'], data['pred'])
            cluster_f1[cluster_id] = metrics['f1_macro']

        return cluster_f1

    def _assess_boundary_health(self, samples: List[dict]) -> float:
        """Assess decision boundary quality"""
        # Simple heuristic: average confidence on predictions
        texts = [s['example'] for s in samples]
        probs = self.model.predict_proba(texts)
        max_probs = np.max(probs, axis=1)
        return float(np.mean(max_probs))

    def _update_data_pools(
            self,
            human_samples: List[dict],
            cf_samples: List[dict],
            training_data: List[dict]
    ):
        """Update labeled and unlabeled pools"""
        # Add to labeled pool
        accepted_samples = [x for x in training_data if x.get('accepted', True)]
        self.labeled_data.extend(accepted_samples)

        # Remove from unlabeled pool
        processed_ids = set(
            [s['id'] for s in human_samples] +
            [s['id'] for s in cf_samples]
        )
        self.unlabeled_data = [
            s for s in self.unlabeled_data
            if s['id'] not in processed_ids
        ]

        # Update scorer with new labeled embeddings
        if accepted_samples:
            texts = [s['example'] for s in accepted_samples]
            new_embeddings = self.embedding_gen.encode(texts)
            self.scorer.update_labeled_embeddings(new_embeddings)

    def _adapt_thresholds(self):
        """Adapt thresholds for next round"""
        if not self.al_config['adaptation']['adjust_thresholds']:
            return

        # Thresholds automatically recompute from percentiles
        # Could add explicit threshold adjustment logic here
        pass

    def _compute_cf_acceptance_rate(self, cf_generated: List[dict]) -> float:
        """Compute CF acceptance rate"""
        if not cf_generated:
            return 1.0
        accepted = sum(1 for x in cf_generated if x.get('accepted', False))
        return accepted / len(cf_generated)

    def _compute_round_costs(
            self,
            human_samples: List[dict],
            cf_samples: List[dict]
    ) -> Dict:
        """Compute annotation costs"""
        budget = self.al_config['budget']

        human_cost = len(human_samples) * budget['cost_human']
        cf_cost = len(cf_samples) * budget['cost_cf']

        return {
            'human_cost': human_cost,
            'cf_cost': cf_cost,
            'total_cost': human_cost + cf_cost
        }

    def _should_stop(self, round_num: int) -> bool:
        """Check stopping criteria"""
        stopping = self.al_config['stopping']

        # Max rounds
        if round_num >= stopping['max_rounds']:
            return True

        # Target F1 reached
        if len(self.round_history) > 0:
            last_f1 = self.round_history[-1]['metrics'].get('f1_macro', 0)
            if last_f1 >= stopping['target_f1']:
                return True

        # No improvement
        if len(self.round_history) >= stopping['patience']:
            recent_f1s = [
                r['metrics'].get('f1_macro', 0)
                for r in self.round_history[-stopping['patience']:]
            ]
            max_improvement = max(recent_f1s) - min(recent_f1s)
            if max_improvement < stopping['min_improvement']:
                return True

        # Unlabeled pool exhausted
        if len(self.unlabeled_data) == 0:
            return True

        return False

    def _print_thresholds(self, thresholds: Dict):
        """Print threshold values"""
        print("  Thresholds:")
        print(f"    τ_h (uncertainty high): {thresholds['tau_h']:.3f}")
        print(f"    τ_l (uncertainty low):  {thresholds['tau_l']:.3f}")
        print(f"    δ   (novelty):          {thresholds['delta']:.3f}")
        print(f"    γ   (feasibility):      {thresholds['gamma']:.3f}")
        print(f"    c*  (coverage):         {thresholds['c_star']:.3f}")
        print(f"    Risk threshold:         {thresholds['risk_threshold']:.3f}")

    def _save_round_checkpoint(self, round_num: int, round_result: Dict):
        """Save round checkpoint"""
        checkpoint_dir = self.config['directories']['checkpoints']
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save round data
        checkpoint_file = f"{checkpoint_dir}/round_{round_num}_checkpoint.json"
        with open(checkpoint_file, 'w') as f:
            # Convert numpy types to Python types
            result_copy = json.loads(
                json.dumps(round_result, default=lambda x: float(x) if isinstance(x, np.floating) else x)
            )
            json.dump(result_copy, f, indent=2)

        # Save model
        model_file = f"{checkpoint_dir}/round_{round_num}_model.pt"
        self.trainer.save_model(self.model, model_file)

    def _compile_results(self) -> Dict:
        """Compile final results"""
        routing_summary = self.router.get_routing_summary()

        # Compute totals
        total_cost = sum(r['costs']['total_cost'] for r in self.round_history)
        final_f1 = self.round_history[-1]['metrics'].get('f1_macro', 0) if self.round_history else 0

        return {
            'rounds': len(self.round_history),
            'final_f1': final_f1,
            'total_labeled': len(self.labeled_data),
            'routing_summary': routing_summary,
            'total_cost': total_cost,
            'round_history': self.round_history,
            'config': self.config
        }

    def _save_results(self, results: Dict):
        """Save final results"""
        output_dir = self.config['directories']['output_data']
        results_file = f"{output_dir}/al_final_results.json"

        with open(results_file, 'w') as f:
            # Handle numpy types
            results_copy = json.loads(
                json.dumps(results, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)
            )
            json.dump(results_copy, f, indent=2)

        print(f"\n✓ Results saved to: {results_file}")


def main():
    """Main execution"""
    print("\n" + "=" * 70)
    print("Script 05: Hybrid Active Learning Loop")
    print("=" * 70)

    # Load configuration
    config = load_config()
    ensure_directories(config)

    # Load data
    dirs = config['directories']
    dataset_config = config['dataset']

    train_path = f"{dirs['input_data']}/{dataset_config['train_file']}"
    test_path = f"{dirs['input_data']}/{dataset_config['test_file']}"

    print(f"\nLoading data...")
    train_df = pd.read_csv(train_path)

    # Split into initial labeled and unlabeled pool
    # Use first N samples as initial labeled set
    n_initial = 100
    initial_labeled = train_df.iloc[:n_initial]
    unlabeled_pool = train_df.iloc[n_initial:]

    print(f"  Initial labeled: {len(initial_labeled)}")
    print(f"  Unlabeled pool: {len(unlabeled_pool)}")

    # Load test data if available
    test_df = None
    if os.path.exists(test_path):
        test_df = pd.read_csv(test_path)
        print(f"  Test set: {len(test_df)}")

    # Initialize and run
    al_system = HybridActiveLearning(config)
    results = al_system.run(initial_labeled, unlabeled_pool, test_df)

    # Print summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Rounds completed: {results['rounds']}")
    print(f"Final F1 score: {results['final_f1']:.4f}")
    print(f"Total labeled: {results['total_labeled']}")
    print(f"Total cost: {results['total_cost']:.2f}")
    print(f"\nRouting summary:")
    for key, value in results['routing_summary'].items():
        print(f"  {key}: {value}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
