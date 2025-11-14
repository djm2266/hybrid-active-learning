#!/usr/bin/env python3
"""
Counterfactual Validator
Validates generated counterfactuals for quality
"""

from typing import Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer, util


class CounterfactualValidator:
    """Validate counterfactual quality"""

    def __init__(self, config: dict):
        self.config = config['counterfactual']['validation']

        # Load similarity model
        embed_config = config['embeddings']
        self.similarity_model = SentenceTransformer(embed_config['model'])

        # Track validation stats for adaptive thresholds
        self.validation_history = []
        self.adaptive_threshold = self.config.get('confidence_threshold', 0.7)

    def validate(
        self,
        original: dict,
        counterfactual: dict,
        model
    ) -> Tuple[bool, Dict]:
        """
        Validate a counterfactual.

        Args:
            original: Original sample
            counterfactual: Generated counterfactual
            model: Trained model

        Returns:
            (is_valid, validation_result)
        """
        checks: Dict = {}

        orig_text = original['example']
        cf_text = counterfactual['example']
        orig_label = original.get('Label', original.get('label'))
        cf_label = counterfactual['Label']

        # ---- 1. Model predictions on original + CF ----
        orig_true_label = orig_label
        orig_pred = model.predict([orig_text])[0]

        pred = model.predict([cf_text])[0]
        probs = model.predict_proba([cf_text])[0]
        max_prob = float(np.max(probs))

        checks['pred_label'] = pred
        checks['cf_label'] = cf_label
        checks['orig_true_label'] = orig_true_label
        checks['orig_pred_label'] = orig_pred
        checks['confidence'] = max_prob

        # ---- 2. Label flip definition ----
        flip_mode = self.config.get('flip_mode', 'hybrid')  # 'target', 'not_original', 'gold_change', 'hybrid'

        if flip_mode == 'not_original':
            # Require CF to leave original decision region
            checks['label_flip'] = (pred != orig_pred)

        elif flip_mode == 'gold_change':
            # Accept if CF gold label differs from original gold label
            checks['label_flip'] = (cf_label != orig_true_label)

        elif flip_mode == 'hybrid':
            # Accept if:
            #  (1) model prediction moved away from its original prediction, OR
            #  (2) CF label equals original true label (within-class augmentation)
            checks['label_flip'] = (pred != orig_pred) or (cf_label == orig_true_label)

        else:  # 'target' - strict: must hit the CF target label
            checks['label_flip'] = (pred == cf_label)

        # ---- 3. Confidence threshold (adaptive) ----
        threshold = self._get_adaptive_threshold()
        checks['confidence_ok'] = max_prob >= threshold
        checks['threshold_used'] = threshold

        # ---- 4. Similarity check ----
        similarity = self._compute_similarity(orig_text, cf_text)
        checks['similarity'] = float(similarity)
        checks['similarity_ok'] = (
            self.config['min_similarity'] <= similarity <= self.config['max_similarity']
        )

        # ---- 5. Label leakage check ----
        if self.config.get('check_label_leakage', False):
            has_leakage = self._check_label_leakage(cf_text, cf_label)
            checks['no_label_leakage'] = not has_leakage
        else:
            checks['no_label_leakage'] = True

        # ---- 6. Overall decision ----
        required_checks = []

        # Always check confidence
        required_checks.append(checks.get('confidence_ok', False))

        # Always check similarity
        required_checks.append(checks.get('similarity_ok', False))

        # Label flip - only if required
        if self.config.get('require_label_flip', True):
            required_checks.append(checks.get('label_flip', False))

        # Label leakage - only if enabled
        if self.config.get('check_label_leakage', False):
            required_checks.append(checks.get('no_label_leakage', True))

        is_valid = all(required_checks)

        if not is_valid:
            checks['reason'] = self._get_rejection_reason(checks)

            # -------- DEBUG OUTPUT FOR REJECTED CFs --------
            print("\n[CF-VALIDATION] REJECTED")
            print(f"  orig_true_label   = {orig_label!r}")
            print(f"  cf_target_label   = {cf_label!r}")
            print(f"  cf_model_pred     = {pred!r}")
            print(f"  confidence        = {max_prob:.3f}")
            print(f"  threshold_used    = {threshold:.3f}")
            print(f"  confidence_ok     = {checks['confidence_ok']}")
            print(f"  similarity        = {similarity:.3f}")
            print(f"  similarity_range  = [{self.config['min_similarity']}, {self.config['max_similarity']}]")
            print(f"  similarity_ok     = {checks['similarity_ok']}")
            print(f"  label_flip        = {checks['label_flip']}")
            print(f"  no_label_leakage  = {checks['no_label_leakage']}")
            print(f"  reason            = {checks['reason']}")
            print("  --- ORIGINAL TEXT ---")
            print(f"  {orig_text[:300]!r}")
            print("  --- COUNTERFACTUAL TEXT ---")
            print(f"  {cf_text[:300]!r}")
            print("------------------------------------------------\n")

        # Track for adaptive threshold
        self.validation_history.append({
            'confidence': max_prob,
            'is_valid': is_valid
        })

        return is_valid, checks

    def _get_adaptive_threshold(self) -> float:
        """
        Get adaptive confidence threshold based on validation history.
        Lowers threshold if rejection rate is too high.
        """
        if len(self.validation_history) < 10:
            # Not enough data, use configured threshold
            return self.config.get('confidence_threshold', 0.7)

        # Look at recent validations
        recent = self.validation_history[-50:]
        acceptance_rate = sum(1 for v in recent if v['is_valid']) / len(recent)
        avg_confidence = np.mean([v['confidence'] for v in recent])

        base_threshold = self.config.get('confidence_threshold', 0.7)

        if acceptance_rate < 0.1 and avg_confidence < 0.5:
            # Very low acceptance, significantly lower threshold
            self.adaptive_threshold = max(0.3, avg_confidence * 0.8)
        elif acceptance_rate < 0.3:
            # Low acceptance, moderately lower threshold
            self.adaptive_threshold = max(0.4, base_threshold * 0.7)
        else:
            # Reasonable acceptance, use base threshold
            self.adaptive_threshold = base_threshold

        return self.adaptive_threshold

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity"""
        embeddings = self.similarity_model.encode([text1, text2])
        similarity = util.cos_sim(embeddings[0], embeddings[1])
        return float(similarity[0][0])

    def _check_label_leakage(self, text: str, label: str) -> bool:
        """Check if label appears in text"""
        text_lower = text.lower()
        label_lower = label.lower()
        return label_lower in text_lower

    def _get_rejection_reason(self, checks: Dict) -> str:
        """Get reason for rejection"""
        if not checks.get('label_flip'):
            return 'no_label_flip'
        if not checks.get('confidence_ok'):
            return 'low_confidence'
        if not checks.get('similarity_ok'):
            return 'similarity_out_of_range'
        if not checks.get('no_label_leakage'):
            return 'label_leakage'
        return 'unknown'
