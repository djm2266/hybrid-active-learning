#!/usr/bin/env python3
"""
Active Learning Router
Implements routing logic from Algorithm 1 Step 3.

Routes samples to:
- Human Oracle (high uncertainty + coverage/novelty, or high risk)
- Counterfactual Generation (medium uncertainty + high feasibility + low risk)
- Defer (low uncertainty or not meeting criteria)
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class RoutingDecision:
    """Container for routing decision"""
    sample_id: str
    route: str  # 'human', 'cf', 'defer'
    scores: Dict[str, float]
    reason: str


class ActiveLearningRouter:
    """
    Routes samples between human oracle and counterfactual generation
    based on multiple criteria.
    """
    
    def __init__(self, config: dict):
        """
        Initialize router with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.al_config = config['active_learning']
        self.knob = self.al_config['knob']['cf_weight']
        self.adaptive = self.al_config['knob']['adaptive']
        
        # Budget tracking
        self.human_budget = self.al_config['budget']['human_budget_per_round']
        self.cf_budget = self.al_config['budget']['cf_budget_per_round']
        self.human_count = 0
        self.cf_count = 0
        
        # Statistics for adaptation
        self.cf_acceptance_rate = 1.0
        self.round_history = []
    
    def route_samples(
        self,
        samples: List[dict],
        scores: Dict[str, np.ndarray],
        thresholds: Dict[str, float]
    ) -> Tuple[List[dict], List[dict], List[dict]]:
        """
        Route samples according to Algorithm 1 Step 3.
        
        Routing logic:
        - IF u(x) >= τ_h AND (c(x) >= c* OR d(x) >= δ) OR r(x) is high
            → Send to Human Oracle
        - ELIF τ_l <= u(x) < τ_h AND g(x) >= γ AND r(x) is low
            → Send to Counterfactual Generation
        - ELSE
            → Defer to next round
        
        Args:
            samples: List of sample dictionaries
            scores: Dictionary of score arrays
            thresholds: Dictionary of threshold values
            
        Returns:
            Tuple of (human_samples, cf_samples, deferred_samples)
        """
        n_samples = len(samples)
        
        # Extract scores and thresholds
        u = scores['uncertainty']
        d = scores['novelty']
        c = scores['coverage']
        g = scores['feasibility']
        r = scores['risk']
        
        tau_h = thresholds['tau_h']
        tau_l = thresholds['tau_l']
        delta = thresholds['delta']
        gamma = thresholds['gamma']
        c_star = thresholds['c_star']
        risk_threshold = thresholds['risk_threshold']
        
        # Initialize routing lists
        human_samples = []
        cf_samples = []
        deferred_samples = []
        decisions = []
        
        # Reset counts for this round
        self.human_count = 0
        self.cf_count = 0
        
        # Apply knob adjustment to thresholds
        adjusted_gamma = self._adjust_feasibility_threshold(gamma)
        
        # Route each sample
        for idx in range(n_samples):
            sample = samples[idx]
            sample_scores = {
                'uncertainty': u[idx],
                'novelty': d[idx],
                'coverage': c[idx],
                'feasibility': g[idx],
                'risk': r[idx]
            }
            
            decision = self._route_single_sample(
                sample, sample_scores, thresholds, adjusted_gamma
            )
            
            decisions.append(decision)
            
            # Add to appropriate list based on routing decision
            if decision.route == 'human':
                if self.human_count < self.human_budget:
                    human_samples.append(sample)
                    self.human_count += 1
                else:
                    # Budget exhausted, defer
                    deferred_samples.append(sample)
                    
            elif decision.route == 'cf':
                if self.cf_count < self.cf_budget:
                    cf_samples.append(sample)
                    self.cf_count += 1
                else:
                    # Budget exhausted, defer
                    deferred_samples.append(sample)
                    
            else:  # defer
                deferred_samples.append(sample)
        
        # Log routing statistics
        self._log_routing_stats(decisions)
        
        return human_samples, cf_samples, deferred_samples
    
    def _route_single_sample(
        self,
        sample: dict,
        scores: Dict[str, float],
        thresholds: Dict[str, float],
        adjusted_gamma: float
    ) -> RoutingDecision:
        """
        Route a single sample according to the algorithm.
        
        Args:
            sample: Sample dictionary
            scores: Scores for this sample
            thresholds: Global thresholds
            adjusted_gamma: Knob-adjusted feasibility threshold
            
        Returns:
            RoutingDecision object
        """
        u = scores['uncertainty']
        d = scores['novelty']
        c = scores['coverage']
        g = scores['feasibility']
        r = scores['risk']
        
        tau_h = thresholds['tau_h']
        tau_l = thresholds['tau_l']
        delta = thresholds['delta']
        c_star = thresholds['c_star']
        risk_threshold = thresholds['risk_threshold']
        
        # Check routing conditions
        
        # Condition 1: High uncertainty + (coverage need OR novelty) OR high risk
        # → Human Oracle
        high_uncertainty = u >= tau_h
        coverage_need = c >= c_star
        high_novelty = d >= delta
        high_risk = r >= risk_threshold
        
        if high_uncertainty and (coverage_need or high_novelty):
            return RoutingDecision(
                sample_id=sample['id'],
                route='human',
                scores=scores,
                reason='high_uncertainty_and_coverage_or_novelty'
            )
        
        if high_risk:
            return RoutingDecision(
                sample_id=sample['id'],
                route='human',
                scores=scores,
                reason='high_risk'
            )
        
        # Condition 2: Medium uncertainty + high feasibility + low risk
        # → Counterfactual Generation
        medium_uncertainty = tau_l <= u < tau_h
        high_feasibility = g >= adjusted_gamma
        low_risk = r < risk_threshold
        
        if medium_uncertainty and high_feasibility and low_risk:
            return RoutingDecision(
                sample_id=sample['id'],
                route='cf',
                scores=scores,
                reason='medium_uncertainty_high_feasibility_low_risk'
            )
        
        # Condition 3: Otherwise → Defer
        return RoutingDecision(
            sample_id=sample['id'],
            route='defer',
            scores=scores,
            reason='low_uncertainty_or_criteria_not_met'
        )
    
    def _adjust_feasibility_threshold(self, gamma: float) -> float:
        """
        Adjust feasibility threshold based on the knob and CF acceptance rate.
        
        Higher knob → lower threshold (more CF routing)
        Lower CF acceptance → higher threshold (more selective)
        
        Args:
            gamma: Base feasibility threshold
            
        Returns:
            Adjusted threshold
        """
        if not self.adaptive:
            # Static adjustment based on knob
            adjustment = (self.knob - 0.5) * 0.2  # ±0.1 adjustment
            return max(0.0, min(1.0, gamma + adjustment))
        
        # Dynamic adjustment based on CF acceptance rate
        if len(self.round_history) > 0:
            # If acceptance rate is low, increase threshold (be more selective)
            target_acceptance = 0.7
            acceptance_gap = self.cf_acceptance_rate - target_acceptance
            adaptive_adjustment = -acceptance_gap * 0.3
        else:
            adaptive_adjustment = 0.0
        
        # Combine knob and adaptive adjustment
        knob_adjustment = (self.knob - 0.5) * 0.2
        total_adjustment = knob_adjustment + adaptive_adjustment
        
        return max(0.0, min(1.0, gamma + total_adjustment))
    
    def update_cf_acceptance_rate(self, acceptance_rate: float):
        """
        Update CF acceptance rate for adaptive routing.
        
        Args:
            acceptance_rate: Fraction of CFs that passed validation
        """
        self.cf_acceptance_rate = acceptance_rate
        
        # Update knob if adaptive mode is enabled
        if self.adaptive:
            self._adapt_knob()
    
    def _adapt_knob(self):
        """
        Adapt the knob based on CF acceptance rate.
        
        If acceptance rate is high → can increase CF routing
        If acceptance rate is low → decrease CF routing
        """
        target_acceptance = 0.7
        tolerance = 0.1
        
        if self.cf_acceptance_rate < target_acceptance - tolerance:
            # Acceptance too low, reduce CF routing
            self.knob = max(0.0, self.knob - 0.05)
        elif self.cf_acceptance_rate > target_acceptance + tolerance:
            # Acceptance high, can increase CF routing
            self.knob = min(1.0, self.knob + 0.05)
    
    def _log_routing_stats(self, decisions: List[RoutingDecision]):
        """
        Log routing statistics for analysis.
        
        Args:
            decisions: List of routing decisions
        """
        total = len(decisions)
        human = sum(1 for d in decisions if d.route == 'human')
        cf = sum(1 for d in decisions if d.route == 'cf')
        deferred = sum(1 for d in decisions if d.route == 'defer')
        
        # Count reasons
        reasons = {}
        for d in decisions:
            reasons[d.reason] = reasons.get(d.reason, 0) + 1
        
        stats = {
            'total': total,
            'human': human,
            'cf': cf,
            'deferred': deferred,
            'human_pct': 100 * human / total if total > 0 else 0,
            'cf_pct': 100 * cf / total if total > 0 else 0,
            'deferred_pct': 100 * deferred / total if total > 0 else 0,
            'reasons': reasons,
            'knob': self.knob,
            'cf_acceptance_rate': self.cf_acceptance_rate
        }
        
        self.round_history.append(stats)
        
        print(f"\n=== Routing Statistics ===")
        print(f"Total samples: {total}")
        print(f"→ Human: {human} ({stats['human_pct']:.1f}%)")
        print(f"→ CF: {cf} ({stats['cf_pct']:.1f}%)")
        print(f"→ Deferred: {deferred} ({stats['deferred_pct']:.1f}%)")
        print(f"Knob setting: {self.knob:.2f}")
        print(f"CF acceptance rate: {self.cf_acceptance_rate:.2f}")
        print(f"\nRouting reasons:")
        for reason, count in reasons.items():
            print(f"  {reason}: {count}")
    
    def enforce_coverage_quotas(
        self,
        human_samples: List[dict],
        cf_samples: List[dict],
        cluster_info: Dict
    ) -> Tuple[List[dict], List[dict]]:
        """
        Enforce minimum samples per cluster.
        
        Args:
            human_samples: Samples routed to human
            cf_samples: Samples routed to CF
            cluster_info: Cluster assignment information
            
        Returns:
            Adjusted (human_samples, cf_samples)
        """
        if not self.al_config['adaptation']['enforce_cluster_quotas']:
            return human_samples, cf_samples
        
        min_per_cluster = self.al_config['adaptation']['min_samples_per_cluster']
        
        # Count samples per cluster
        cluster_counts = {}
        for sample in human_samples + cf_samples:
            cluster_id = sample.get('cluster_id')
            if cluster_id is not None:
                cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
        
        # Identify underrepresented clusters
        all_clusters = cluster_info.get('cluster_sizes', {}).keys()
        underrep_clusters = [
            c for c in all_clusters
            if cluster_counts.get(c, 0) < min_per_cluster
        ]
        
        # If there are underrepresented clusters, prioritize them
        # This could involve re-routing some deferred samples
        # For now, just log the information
        if underrep_clusters:
            print(f"\nWARNING: Underrepresented clusters: {underrep_clusters}")
            print(f"Consider adding more samples from these clusters")
        
        return human_samples, cf_samples
    
    def get_routing_summary(self) -> Dict:
        """
        Get summary of routing across all rounds.
        
        Returns:
            Dictionary with routing statistics
        """
        if not self.round_history:
            return {}
        
        total_human = sum(r['human'] for r in self.round_history)
        total_cf = sum(r['cf'] for r in self.round_history)
        total_deferred = sum(r['deferred'] for r in self.round_history)
        total = total_human + total_cf + total_deferred
        
        return {
            'total_samples': total,
            'total_human': total_human,
            'total_cf': total_cf,
            'total_deferred': total_deferred,
            'human_pct': 100 * total_human / total if total > 0 else 0,
            'cf_pct': 100 * total_cf / total if total > 0 else 0,
            'deferred_pct': 100 * total_deferred / total if total > 0 else 0,
            'avg_knob': np.mean([r['knob'] for r in self.round_history]),
            'avg_cf_acceptance': np.mean([r['cf_acceptance_rate'] for r in self.round_history]),
            'rounds': len(self.round_history)
        }
