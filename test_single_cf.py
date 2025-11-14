#!/usr/bin/env python3
"""
Test Counterfactual Generation on Single Row
Shows exactly what Qwen generates for one example
"""

import sys

sys.path.insert(0, 'src')

from utils import load_config, get_llm_provider
import pandas as pd
import random


def main():
    print("=" * 70)
    print("SINGLE ROW COUNTERFACTUAL GENERATION TEST")
    print("=" * 70)

    # Load config
    config = load_config()

    print(f"\n📋 LLM Provider: {config['llm']['provider']}")
    if config['llm']['provider'] == 'ollama':
        print(f"   Model: {config['llm']['ollama']['model']}")

    # Load LLM
    print("\n🤖 Initializing LLM...")
    try:
        llm = get_llm_provider(config)
        print("   ✓ LLM loaded successfully")
    except Exception as e:
        print(f"   ✗ Error loading LLM: {e}")
        print("\n   Make sure Ollama is running:")
        print("   1. Open terminal: ollama serve")
        print("   2. Pull model: ollama pull qwen2.5:7b")
        return

    # Load data
    print("\n📊 Loading data...")
    data_path = f"{config['directories']['input_data']}/{config['dataset']['train_file']}"
    try:
        df = pd.read_csv(data_path)
        print(f"   ✓ Loaded {len(df)} rows")
    except Exception as e:
        print(f"   ✗ Error loading data: {e}")
        return

    # Get column names
    text_col = config['dataset']['text_column']
    label_col = config['dataset']['label_column']

    # Pick a random row (or first row)
    row_idx = random.randint(0, min(10, len(df) - 1))  # First 10 rows
    row = df.iloc[row_idx]

    original_text = row[text_col]
    original_label = row[label_col]

    # Get all possible labels
    all_labels = df[label_col].unique().tolist()

    # Skip problematic abstract labels
    problematic_labels = ['none', 'other', 'unknown', 'misc', 'n/a']

    # Pick a different target label (avoid abstract ones if possible)
    possible_targets = [
        l for l in all_labels
        if l != original_label and l.lower() not in problematic_labels
    ]

    # If no good targets, use any different label
    if not possible_targets:
        possible_targets = [l for l in all_labels if l != original_label]

    if not possible_targets:
        print("   ✗ Only one label in dataset!")
        return

    target_label = random.choice(possible_targets)

    print(f"\n{'=' * 70}")
    print("SELECTED EXAMPLE")
    print(f"{'=' * 70}")
    print(f"\n📝 Original Text:")
    print(f"   {original_text}")
    print(f"\n🏷️  Original Label: {original_label}")
    print(f"🎯 Target Label: {target_label}")
    print(f"\n{'=' * 70}")

    # Construct the prompt
    print("\n📤 PROMPT SENT TO QWEN:")
    print("-" * 70)

    messages = [
        {
            "role": "system",
            "content": "You are a semantic editor. You transform the meaning of sentences by replacing topic-specific words while keeping the sentence structure."
        },
        {
            "role": "user",
            "content": f"""Transform this sentence from being about '{original_label}' to being about '{target_label}'.

Original sentence: {original_text}

TRANSFORMATION RULES:
1. Identify ALL words related to '{original_label}' (nouns, verbs, adjectives)
2. Replace them with semantically appropriate words for '{target_label}'
3. Do NOT include the literal word '{target_label}' in your output
4. Do NOT include words from '{original_label}' in your output
5. Keep the overall sentence structure similar
6. Output ONLY the final transformed sentence - no explanations or labels

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

    # Print the actual prompt
    for msg in messages:
        print(f"\n[{msg['role'].upper()}]")
        print(msg['content'])

    print("\n" + "-" * 70)
    print("\n⏳ Generating counterfactual with Qwen...")
    print("   (This may take 5-30 seconds...)\n")

    # Generate
    try:
        llm_config = config['llm']['models']['counterfactual_generation']
        response = llm.chat_completion(
            messages=messages,
            temperature=llm_config['temperature'],
            max_tokens=llm_config['max_tokens']
        )

        # Clean response
        counterfactual = response.strip().strip('"\'')

        print("=" * 70)
        print("✅ GENERATED COUNTERFACTUAL")
        print("=" * 70)
        print(f"\n{counterfactual}")
        print("\n" + "=" * 70)

        # Analysis
        print("\n📊 ANALYSIS:")
        print("-" * 70)

        # Check length
        orig_words = len(original_text.split())
        cf_words = len(counterfactual.split())
        words_changed = abs(orig_words - cf_words)

        print(f"Original length: {orig_words} words")
        print(f"CF length: {cf_words} words")
        print(f"Difference: {words_changed} words")

        # Check if target label appears
        if target_label.lower() in counterfactual.lower():
            print(f"\n⚠️  WARNING: Target label '{target_label}' appears in CF (label leakage)")
        else:
            print(f"\n✓ No label leakage ('{target_label}' not in CF)")

        # Similarity check (simple word overlap)
        orig_words_set = set(original_text.lower().split())
        cf_words_set = set(counterfactual.lower().split())
        overlap = len(orig_words_set & cf_words_set)
        total = len(orig_words_set | cf_words_set)
        similarity = overlap / total if total > 0 else 0

        print(f"\nWord overlap: {overlap}/{len(orig_words_set)} original words kept")
        print(f"Jaccard similarity: {similarity:.2%}")

        if similarity > 0.7:
            print("✓ Good similarity (minimal changes)")
        elif similarity > 0.4:
            print("⚠️  Moderate similarity (some changes)")
        else:
            print("✗ Low similarity (major changes)")

        print("\n" + "=" * 70)
        print("💡 INTERPRETATION")
        print("=" * 70)

        if similarity > 0.6 and target_label.lower() not in counterfactual.lower():
            print("\n✅ GOOD COUNTERFACTUAL")
            print("   - Minimal changes from original")
            print("   - No label leakage")
            print("   - Should express target concept")
        elif similarity < 0.3:
            print("\n⚠️  POOR COUNTERFACTUAL")
            print("   - Too different from original")
            print("   - May not preserve context")
            print("   - Consider tuning prompt or using better model")
        else:
            print("\n➖ OKAY COUNTERFACTUAL")
            print("   - Moderate changes")
            print("   - Might work depending on validation thresholds")

    except Exception as e:
        print(f"\n✗ Error generating counterfactual: {e}")
        print("\nPossible issues:")
        print("1. Ollama not running (run: ollama serve)")
        print("2. Model not downloaded (run: ollama pull qwen2.5:7b)")
        print("3. Connection error")
        return

    print("\n" + "=" * 70)
    print("WANT TO TRY ANOTHER?")
    print("=" * 70)
    print("\nJust run this script again to test a different random example!")
    print("Or modify row_idx in the script to test a specific row.")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()