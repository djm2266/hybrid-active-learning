#!/usr/bin/env python3
"""
Human Oracle Interface
In this variant, the "human" oracle always uses the gold label
that already exists on the sample (no CLI/web/file interaction).
"""

from typing import Optional, Dict


class HumanOracle:
    """Interface for obtaining human annotations (gold-label only version)."""

    def __init__(self, config: dict):
        # Keep config for compatibility, but interface_type is effectively ignored
        self.config = config.get("human_oracle", {})
        self.interface_type = self.config.get("interface", "gold")
        self.annotation_count = 0

    def get_label(self, sample: Dict) -> Optional[str]:
        """
        Get label for a sample.

        This implementation *always* uses the existing gold label on the sample,
        if present, and never asks the user for input.

        Args:
            sample: Sample to "annotate" (must contain a label field).

        Returns:
            Label string.

        Raises:
            KeyError if no label-like field is found.
        """
        # Support both "label" and "Label" to match typical dataset conventions
        if "label" in sample:
            gold_label = sample["label"]
        elif "Label" in sample:
            gold_label = sample["Label"]
        else:
            # You can change this to return None instead if you prefer silent failure
            raise KeyError(
                f"HumanOracle expected a gold label in sample, but found neither "
                f"'label' nor 'Label'. Sample keys: {list(sample.keys())}"
            )

        self.annotation_count += 1
        # Optional debug print
        print(f"[HUMAN-ORACLE] Using gold label for sample {sample.get('id')}: {gold_label}")
        return gold_label

