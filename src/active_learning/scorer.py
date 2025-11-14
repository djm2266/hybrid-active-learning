#!/usr/bin/env python3
"""
Active Learning Scorer
Implements all scoring functions from Algorithm 1:
- u(x): Uncertainty (normalized entropy)
- d(x): Novelty (distance to labeled set)
- c(x): Coverage (cluster-based need)
- g(x): CF-feasibility
- r(x): Risk assessment
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.spatial.distance import cdist
from scipy.stats import entropy
from sklearn.cluster import KMeans
import torch


class ActiveLearningScorer:
    """
    Computes all scores needed for hybrid active learning routing.
    """
    
    def __init__(self, config: dict):
        """
        Initialize scorer with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.scoring_config = config['scoring']
        self.al_config = config['active_learning']
        
        # Clustering for coverage
        self.num_clusters = self.scoring_config['coverage']['num_clusters']
        self.clusterer = None
        self.cluster_assignments = None
        self.cluster_sizes = {}
        
        # For novelty computation
        self.labeled_embeddings = None
        
    def compute_all_scores(
        self,
        samples: List[dict],
        model,
        embeddings: np.ndarray,
        labeled_embeddings: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Compute all scoring functions for a batch of samples.
        
        Args:
            samples: List of sample dictionaries with text
            model: Trained model for predictions
            embeddings: Embeddings for all samples (N x D)
            labeled_embeddings: Embeddings of labeled set (M x D)
            
        Returns:
            Dictionary with scores: {
                'uncertainty': array of shape (N,),
                'novelty': array of shape (N,),
                'coverage': array of shape (N,),
                'feasibility': array of shape (N,),
                'risk': array of shape (N,)
            }
        """
        n_samples = len(samples)
        
        # Compute individual scores
        uncertainty = self.compute_uncertainty(samples, model)
        novelty = self.compute_novelty(embeddings, labeled_embeddings)
        coverage = self.compute_coverage(embeddings)
        feasibility = self.compute_feasibility(samples, model)
        risk = self.compute_risk(samples, model, feasibility)
        
        return {
            'uncertainty': uncertainty,
            'novelty': novelty,
            'coverage': coverage,
            'feasibility': feasibility,
            'risk': risk
        }
    
    def compute_uncertainty(self, samples: List[dict], model) -> np.ndarray:
        """
        Compute uncertainty score u(x) = -Σ p_k log(p_k) / log(K)
        
        Normalized entropy over K classes.
        
        Args:
            samples: List of samples
            model: Trained model
            
        Returns:
            Array of uncertainty scores (N,)
        """
        # Get model predictions
        texts = [s['example'] for s in samples]
        probs = model.predict_proba(texts)  # Shape: (N, K)
        
        # Compute entropy
        method = self.scoring_config['uncertainty']['method']
        
        if method == 'entropy':
            # Normalized entropy: -Σ p_k log(p_k) / log(K)
            K = probs.shape[1]
            eps = 1e-10
            entropies = -np.sum(probs * np.log(probs + eps), axis=1)
            normalized = entropies / np.log(K)
            return normalized
            
        elif method == 'margin':
            # Margin sampling: 1 - (p1 - p2)
            sorted_probs = np.sort(probs, axis=1)
            margins = sorted_probs[:, -1] - sorted_probs[:, -2]
            return 1 - margins
            
        elif method == 'least_confident':
            # Least confident: 1 - max(p)
            return 1 - np.max(probs, axis=1)
            
        else:
            raise ValueError(f"Unknown uncertainty method: {method}")
    
    def compute_novelty(
        self,
        embeddings: np.ndarray,
        labeled_embeddings: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Compute novelty score d(x) = min_{z_i ∈ L} ||z_x - z_i||_2
        
        Distance to nearest labeled example.
        
        Args:
            embeddings: Embeddings for unlabeled samples (N, D)
            labeled_embeddings: Embeddings for labeled samples (M, D)
            
        Returns:
            Array of novelty scores (N,)
        """
        if labeled_embeddings is None or len(labeled_embeddings) == 0:
            # No labeled data yet, all samples equally novel
            return np.ones(len(embeddings))
        
        method = self.scoring_config['novelty']['method']
        
        # Compute pairwise distances
        if method == 'euclidean':
            distances = cdist(embeddings, labeled_embeddings, metric='euclidean')
        elif method == 'cosine':
            distances = cdist(embeddings, labeled_embeddings, metric='cosine')
        elif method == 'manhattan':
            distances = cdist(embeddings, labeled_embeddings, metric='cityblock')
        else:
            raise ValueError(f"Unknown novelty method: {method}")
        
        # Take minimum distance to any labeled point
        min_distances = np.min(distances, axis=1)
        
        return min_distances
    
    def compute_coverage(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute coverage score c(x) = 1 - |C_j| / max_i |C_i|
        
        Inverse cluster size (prioritize underrepresented clusters).
        
        Args:
            embeddings: Embeddings for samples (N, D)
            
        Returns:
            Array of coverage scores (N,)
        """
        # Perform clustering if not done yet
        if self.clusterer is None:
            method = self.scoring_config['coverage']['clustering_method']
            
            if method == 'kmeans':
                self.clusterer = KMeans(
                    n_clusters=self.num_clusters,
                    random_state=self.config['processing']['seed']
                )
                self.cluster_assignments = self.clusterer.fit_predict(embeddings)
            else:
                raise ValueError(f"Unknown clustering method: {method}")
        else:
            # Assign to existing clusters
            self.cluster_assignments = self.clusterer.predict(embeddings)
        
        # Count cluster sizes
        unique, counts = np.unique(self.cluster_assignments, return_counts=True)
        self.cluster_sizes = dict(zip(unique, counts))
        max_size = max(counts)
        
        # Compute coverage score for each sample
        coverage_scores = np.zeros(len(embeddings))
        for idx, cluster_id in enumerate(self.cluster_assignments):
            cluster_size = self.cluster_sizes[cluster_id]
            # Higher score for smaller clusters (need more coverage)
            coverage_scores[idx] = 1 - (cluster_size / max_size)
        
        return coverage_scores
    
    def compute_feasibility(
        self,
        samples: List[dict],
        model
    ) -> np.ndarray:
        """
        Compute CF-feasibility score:
        g(x) = 0.5 * (1 - tanh(0.05(len(x) - L_0))) + 0.5 * max_i a_i / Σ a_i
        
        Based on text length and attention weights.
        
        Args:
            samples: List of samples with text
            model: Model (for attention if available)
            
        Returns:
            Array of feasibility scores (N,)
        """
        config = self.scoring_config['feasibility']
        L_0 = config['target_length']
        length_weight = config['length_weight']
        attention_weight = config['attention_weight']
        
        n_samples = len(samples)
        feasibility_scores = np.zeros(n_samples)
        
        for idx, sample in enumerate(samples):
            text = sample['example']
            text_len = len(text.split())
            
            # Length-based component
            length_penalty = 1 - np.tanh(0.05 * (text_len - L_0))
            length_score = 0.5 * (1 + length_penalty)
            
            # Attention-based component (if available)
            attention_score = 0.5  # Default
            if hasattr(model, 'get_attention_weights'):
                try:
                    attention_weights = model.get_attention_weights(text)
                    # Use max attention / sum attention as focus measure
                    if attention_weights is not None and len(attention_weights) > 0:
                        max_attn = np.max(attention_weights)
                        sum_attn = np.sum(attention_weights)
                        attention_score = max_attn / sum_attn if sum_attn > 0 else 0.5
                except:
                    pass  # Fall back to default
            
            # Combine components
            feasibility_scores[idx] = (
                length_weight * length_score +
                attention_weight * attention_score
            )
        
        return feasibility_scores
    
    def compute_risk(
        self,
        samples: List[dict],
        model,
        feasibility: np.ndarray
    ) -> np.ndarray:
        """
        Compute risk score r(x) = P_unsafe(x) + α(1 - g(x))
        
        Combines safety score and infeasibility penalty.
        
        Args:
            samples: List of samples
            model: Model for classification
            feasibility: Pre-computed feasibility scores
            
        Returns:
            Array of risk scores (N,)
        """
        config = self.scoring_config['risk']
        alpha = config['alpha']
        safety_weight = config['safety_weight']
        infeasibility_weight = config['infeasibility_weight']
        
        n_samples = len(samples)
        risk_scores = np.zeros(n_samples)
        
        # Safety component
        safety_scores = np.zeros(n_samples)
        if config['safety_model'] is not None:
            # Use external safety classifier if available
            safety_model = config['safety_model']
            texts = [s['example'] for s in samples]
            safety_scores = safety_model.predict_unsafe_proba(texts)
        else:
            # Default: no safety concerns
            safety_scores = np.zeros(n_samples)
        
        # Infeasibility penalty
        infeasibility = 1 - feasibility
        
        # Combine
        risk_scores = (
            safety_weight * safety_scores +
            infeasibility_weight * alpha * infeasibility
        )
        
        return risk_scores
    
    def compute_thresholds(
        self,
        scores: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Compute dynamic thresholds based on percentiles.
        
        From Algorithm 1:
        - τ_h: uncertainty high (80th percentile)
        - τ_l: uncertainty low (40th percentile)
        - δ: novelty (70th percentile)
        - γ: feasibility (60th percentile)
        - c*: coverage (70th percentile)
        
        Args:
            scores: Dictionary of score arrays
            
        Returns:
            Dictionary of threshold values
        """
        thresholds = self.al_config['thresholds']
        
        return {
            'tau_h': np.percentile(
                scores['uncertainty'],
                thresholds['uncertainty_high_percentile']
            ),
            'tau_l': np.percentile(
                scores['uncertainty'],
                thresholds['uncertainty_low_percentile']
            ),
            'delta': np.percentile(
                scores['novelty'],
                thresholds['novelty_percentile']
            ),
            'gamma': np.percentile(
                scores['feasibility'],
                thresholds['feasibility_percentile']
            ),
            'c_star': np.percentile(
                scores['coverage'],
                thresholds['coverage_percentile']
            ),
            'risk_threshold': np.percentile(
                scores['risk'],
                thresholds['risk_percentile']
            )
        }
    
    def update_labeled_embeddings(self, labeled_embeddings: np.ndarray):
        """
        Update the labeled set embeddings for novelty computation.
        
        Args:
            labeled_embeddings: New labeled embeddings
        """
        if self.labeled_embeddings is None:
            self.labeled_embeddings = labeled_embeddings
        else:
            self.labeled_embeddings = np.vstack([
                self.labeled_embeddings,
                labeled_embeddings
            ])
    
    def get_cluster_info(self) -> Dict:
        """
        Get current clustering information.
        
        Returns:
            Dictionary with cluster statistics
        """
        if self.cluster_assignments is None:
            return {}
        
        return {
            'num_clusters': self.num_clusters,
            'cluster_sizes': self.cluster_sizes,
            'assignments': self.cluster_assignments
        }
