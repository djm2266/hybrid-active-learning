#!/usr/bin/env python3
"""
Counterfactual Generator
Generates counterfactual examples using LLM
"""

from typing import Dict, Optional, List
import numpy as np  # for debug on model probs


class CounterfactualGenerator:
    """Generate counterfactual examples"""

    def __init__(self, config: dict, llm_provider):
        self.config = config['counterfactual']['generation']
        self.llm_provider = llm_provider
        self.llm_config = config['llm']['models']['counterfactual_generation']

    def generate(self, sample: dict, model) -> Optional[Dict]:
        """
        Generate counterfactual for a sample.

        Args:
            sample: Original sample with text and predicted label
            model: Trained model for getting predictions

        Returns:
            Counterfactual dict or None if generation failed
        """
        text = sample['example']

        # Get model prediction on the original text
        predictions = model.predict([text])
        probs = model.predict_proba([text])
        original_label = predictions[0]
        original_conf = float(np.max(probs))

        # Get target label (flip to different class)
        target_label = self._get_target_label(original_label, model)

        # Generate candidate phrases first (simplified)
        candidate_phrases = self._generate_candidate_phrases(
            text, original_label, target_label
        )

        if not candidate_phrases:
            print("[CF-GEN] No candidate phrases generated; skipping.")
            return None

        # Generate full counterfactual
        messages = self._construct_generation_prompt(
            text, original_label, target_label, candidate_phrases
        )

        try:
            response = self.llm_provider.chat_completion(
                messages=messages,
                temperature=self.llm_config['temperature'],
                max_tokens=self.llm_config['max_tokens']
            )

            counterfactual_text = (response or "").strip().strip('"\'')

            # If LLM gave us nothing useful, bail
            if not counterfactual_text:
                print("[CF-GEN] Empty counterfactual text from LLM; skipping.")
                return None

            # --- DEBUG: look at model behavior on the CF ---
            cf_pred = model.predict([counterfactual_text])[0]
            cf_probs = model.predict_proba([counterfactual_text])[0]
            cf_conf = float(np.max(cf_probs))

            print("\n[CF-GEN] Generated counterfactual")
            print(f"  original_label      = {original_label!r} (conf={original_conf:.3f})")
            print(f"  target_label        = {target_label!r}")
            print(f"  cf_model_pred_label = {cf_pred!r} (conf={cf_conf:.3f})")
            print("  --- original text ---")
            print(f"  {text}")
            print("  --- counterfactual text ---")
            print(f"  {counterfactual_text}")
            print("  --- candidate_phrases ---")
            print(f"  {candidate_phrases}")
            print("")

            return {
                'example': counterfactual_text,
                # This is the *intended* label for the CF
                'Label': target_label,
                'original_text': text,
                'original_label': original_label,
                'candidate_phrases': candidate_phrases,
                # Optional debug info downstream
                'cf_model_pred_label': cf_pred,
                'cf_model_pred_conf': cf_conf,
            }

        except Exception as e:
            print(f"  CF generation error: {str(e)[:80]}")
            return None

    def _get_target_label(self, original_label: str, model) -> str:
        """Get target label for flip"""
        all_labels = model.classes_
        # Simple: pick a different class
        for label in all_labels:
            if label != original_label:
                return label
        return original_label

    def _generate_candidate_phrases(
        self,
        text: str,
        original_label: str,
        target_label: str
    ) -> List[str]:
        """Generate candidate phrases (simplified version)"""
        # This is a simplified version - full version uses an LLM in Script 02
        # For now, return placeholder
        return [f"phrase_for_{target_label}"]

    def _construct_generation_prompt(
        self,
        text: str,
        original_label: str,
        target_label: str,
        candidate_phrases: List[str]
    ) -> List[Dict]:
        """Construct prompt for CF generation - domain agnostic"""

        # Optional: Check if domain hints are provided in config
        domain_hints = self.config.get('domain_hints', {})
        hint = ""

        if domain_hints:
            target_hint = domain_hints.get(target_label, "")
            if target_hint:
                hint = f"\n\nHint for '{target_label}': {target_hint}"

        return [
            {
                "role": "system",
                "content": "You are a semantic editor. You transform the meaning of sentences by replacing topic-specific words while keeping the sentence structure."
            },
            {
                "role": "user",
                "content": f"""Transform this sentence from being about '{original_label}' to being about '{target_label}'.

Original sentence: {text}

TRANSFORMATION RULES:
1. Identify ALL words related to '{original_label}' (nouns, verbs, adjectives)
2. Replace them with semantically appropriate words for '{target_label}'
3. Do NOT include the literal word '{target_label}' in your output
4. Do NOT include words from '{original_label}' in your output
5. Keep the overall sentence structure similar
6. Output ONLY the final transformed sentence - no explanations or labels{hint}

EXAMPLES showing complete semantic transformation:

Example 1 - Lists → Play (media):
Original: "add jazz to my workout list"
Transformed: "start jazz on my music app"
(replaced 'add...list' with 'start...app')

Example 2 - Email → Calendar:
Original: "send a message to John about dinner"
Transformed: "schedule a meeting with John about dinner"
(replaced 'send message' with 'schedule meeting')

Example 3 - Audio → Weather:
Original: "turn up the volume on my speaker"
Transformed: "check the forecast on my phone"
(replaced 'turn up volume on speaker' with 'check forecast on phone')

Example 4 - Shopping → Navigation:
Original: "add milk to my cart"
Transformed: "add home to my route"
(replaced 'milk...cart' with 'home...route')

KEY PRINCIPLE: The words must change to match the new topic, but the sentence should still make grammatical sense.

Now transform the sentence to be about '{target_label}':"""
            }
        ]
