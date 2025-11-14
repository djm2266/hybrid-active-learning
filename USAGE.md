# Hybrid Active Learning - Usage Guide

## Quick Start (5 Minutes)

### 1. Setup Environment

```bash
# Clone or extract the repository
cd hybrid-active-learning

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Check setup
python check_setup.py
```

### 2. Configure LLM Provider

**Option A: Ollama (Recommended for local/free)**
```bash
# Install from https://ollama.ai
ollama serve  # In one terminal
ollama pull qwen2.5:7b  # In another terminal
```

**Option B: Gemini**
```yaml
# Edit config/config.yaml
llm:
  provider: gemini
  gemini:
    api_key: YOUR_KEY_HERE  # Get from https://makersuite.google.com
```

### 3. Run Pipeline

```bash
# Using example data
python run_pipeline.py

# With your own data
# 1. Add your data to data/input/train.csv
# 2. python run_pipeline.py
```

## Understanding the Algorithm

The system implements a hybrid approach that intelligently routes samples:

```
                    ┌─────────────────────┐
                    │  Unlabeled Pool (U) │
                    └──────────┬──────────┘
                               │
                ┌──────────────▼─────────────┐
                │  Step 1: Score Samples     │
                │  • Uncertainty (entropy)    │
                │  • Novelty (distance)       │
                │  • Coverage (clusters)      │
                │  • Feasibility (length+attn)│
                │  • Risk (safety+infeas)     │
                └──────────────┬─────────────┘
                               │
                ┌──────────────▼─────────────┐
                │  Step 2: Dynamic Thresholds│
                │  τh, τl, δ, γ, c* (percentiles)
                └──────────────┬─────────────┘
                               │
                ┌──────────────▼─────────────┐
                │  Step 3: Route Samples     │
                │                            │
    ┌───────────┴──────────┬────────────────┴───────────┐
    │                      │                            │
┌───▼──────────┐   ┌───▼──────────┐      ┌────▼─────────┐
│   HUMAN      │   │     CF       │      │   DEFER      │
│  high u      │   │  medium u    │      │  low u       │
│  + (cov/nov) │   │  + high g    │      │  or criteria │
│  OR high r   │   │  + low r     │      │  not met     │
└───┬──────────┘   └───┬──────────┘      └────┬─────────┘
    │                  │                      │
    │ weight=1.0       │ weight=0.7           │ (next round)
    │                  │                      │
    └──────────────────┴──────────────────────┘
                       │
            ┌──────────▼───────────┐
            │ Step 4: Obtain Labels│
            │ Step 5: Train Model  │
            │ Step 6: Adapt        │
            └──────────────────────┘
```

## Configuration Knobs

### Primary Knob: CF Weight (0.0 to 1.0)

```yaml
active_learning:
  knob:
    cf_weight: 0.5  # ← THIS IS THE MAIN KNOB
```

**What it does:**
- `0.0` = Pure human oracle (traditional AL)
- `0.3` = Mostly human, CF for easy cases
- `0.5` = Balanced (default)
- `0.7` = Mostly CF, human for hard cases
- `1.0` = Maximum CF usage

**How it works:**
Adjusts the feasibility threshold γ:
- Higher knob → lower threshold → more CF routing
- Lower knob → higher threshold → more human routing

### Budget Knobs

```yaml
budget:
  samples_per_round: 100      # Total per round
  human_budget_per_round: 50  # Max human/round
  cf_budget_per_round: 100    # Max CF/round
  total_rounds: 10            # Max rounds
```

### Threshold Knobs (Percentiles)

```yaml
thresholds:
  uncertainty_high_percentile: 80  # Top 20% → human likely
  uncertainty_low_percentile: 40   # Bottom 60% excluded
  novelty_percentile: 70           # Top 30% → prefer human
  feasibility_percentile: 60       # Top 40% → CF eligible
  coverage_percentile: 70          # Top 30% need coverage
```

**Tuning advice:**
- Increase `uncertainty_high` → stricter human routing
- Increase `feasibility` → more selective CF
- Increase `coverage` → focus on rare clusters

### Sample Weight Knobs

```yaml
sample_weights:
  human_labeled: 1.0   # Full trust
  cf_generated: 0.7    # Lower trust
  cf_validated: 0.85   # If human-verified
```

## Typical Workflows

### Workflow 1: Pure Human Oracle (Baseline)

```yaml
knob:
  cf_weight: 0.0  # No CF
budget:
  human_budget_per_round: 100
  cf_budget_per_round: 0
```

```bash
python run_pipeline.py --config config/baseline_human.yaml
```

### Workflow 2: Pure Counterfactual

```yaml
knob:
  cf_weight: 1.0  # Maximum CF
budget:
  human_budget_per_round: 0
  cf_budget_per_round: 200
```

```bash
python run_pipeline.py --config config/max_cf.yaml
```

### Workflow 3: Hybrid with Adaptation

```yaml
knob:
  cf_weight: 0.5
  adaptive: true  # Adjust based on CF quality
```

The system will:
1. Start with 50/50 split
2. Monitor CF acceptance rate
3. Increase CF if acceptance > 80%
4. Decrease CF if acceptance < 60%

### Workflow 4: Budget-Constrained

```yaml
budget:
  total_human_budget: 200  # Total across ALL rounds
  cost_human: 1.0
  cost_cf: 0.1  # CF is 10x cheaper
```

The system prioritizes cost-effective labeling while maintaining quality.

## Experiment Examples

### Experiment 1: Compare Strategies

```bash
# Run with different knob settings
for weight in 0.0 0.3 0.5 0.7 1.0; do
  sed -i "s/cf_weight: .*/cf_weight: $weight/" config/config.yaml
  python run_pipeline.py
  mv data/output/al_final_results.json results/cf_${weight}.json
done

# Analyze results
python notebooks/compare_strategies.py
```

### Experiment 2: Threshold Sensitivity

```bash
# Test different uncertainty thresholds
for threshold in 70 75 80 85 90; do
  sed -i "s/uncertainty_high_percentile: .*/uncertainty_high_percentile: $threshold/" config/config.yaml
  python run_pipeline.py
done
```

### Experiment 3: Budget Analysis

```bash
# Test different budget allocations
python run_pipeline.py --config config/budget_low.yaml
python run_pipeline.py --config config/budget_medium.yaml
python run_pipeline.py --config config/budget_high.yaml
```

## Monitoring and Analysis

### During Training

The system outputs:

```
=== Routing Statistics ===
Total samples: 100
→ Human: 35 (35.0%)
→ CF: 45 (45.0%)
→ Deferred: 20 (20.0%)
Knob setting: 0.50
CF acceptance rate: 0.73

Routing reasons:
  high_uncertainty_and_coverage_or_novelty: 25
  high_risk: 10
  medium_uncertainty_high_feasibility_low_risk: 45
  low_uncertainty_or_criteria_not_met: 20
```

### After Training

Check `data/output/al_final_results.json`:

```json
{
  "rounds": 10,
  "final_f1": 0.8734,
  "total_labeled": 567,
  "routing_summary": {
    "total_human": 234,
    "total_cf": 333,
    "human_pct": 41.3,
    "cf_pct": 58.7,
    "avg_cf_acceptance": 0.76
  },
  "total_cost": 267.3
}
```

## Advanced Topics

### Custom Scoring Functions

```python
# In src/active_learning/scorer.py

def custom_uncertainty(self, samples, model):
    """Your custom uncertainty metric"""
    # Implement your logic
    return uncertainty_scores
```

### Custom Routing Logic

```python
# In src/active_learning/router.py

def _route_single_sample(self, sample, scores, thresholds):
    """Implement custom routing rules"""
    # Your logic here
    return RoutingDecision(...)
```

### Integration with Existing Systems

```python
# Use as a library
from src.active_learning_loop import HybridActiveLearning
from utils import load_config

config = load_config('my_config.yaml')
al = HybridActiveLearning(config)

# Integrate with your data pipeline
results = al.run(
    initial_labeled=my_labeled_df,
    unlabeled_pool=my_unlabeled_df,
    test_data=my_test_df
)
```

## Troubleshooting

### Issue: Low CF Acceptance Rate

**Symptom:** CF acceptance < 50%

**Solutions:**
1. Increase `feasibility_percentile` (be more selective)
2. Use better LLM model (e.g., GPT-4 vs GPT-3.5)
3. Adjust `similarity_threshold` in validation
4. Decrease `cf_weight` (use less CF)

### Issue: Imbalanced Routing

**Symptom:** 90% human or 90% CF

**Solutions:**
1. Check your `cf_weight` setting
2. Review threshold percentiles
3. Enable adaptive mode: `adaptive: true`
4. Check budget constraints

### Issue: Memory Errors

**Solutions:**
1. Reduce `batch_size` in config
2. Reduce `samples_per_round`
3. Use smaller embedding model
4. Enable checkpointing and resume

### Issue: API Quota Exhausted

**Solutions:**
1. The system auto-saves checkpoints every 50 samples
2. Just re-run - it will resume from checkpoint
3. Reduce `samples_per_round`
4. Increase `rate_limiting.requests_per_minute`

## Performance Tips

### For Speed:
- Use local Ollama instead of API
- Reduce embedding dimensions
- Decrease `samples_per_round`
- Use CPU for embeddings if GPU busy

### For Quality:
- Increase `cf_budget` for more options
- Use human validation for critical samples
- Enable `double_annotation`
- Increase initial labeled set size

### For Cost:
- Start with high `cf_weight`
- Lower `human_budget_per_round`
- Enable adaptive mode to optimize
- Use checkpointing to avoid re-runs

## Citation

```bibtex
@software{hybrid_active_learning_2024,
  title = {Hybrid Active Learning with Counterfactuals and Human Oracle},
  year = {2024},
  author = {Your Name},
  url = {https://github.com/yourusername/hybrid-active-learning}
}
```

## Support

- Documentation: See README.md
- Issues: GitHub Issues
- Examples: See notebooks/ directory
- Tests: Run `pytest tests/`

## Next Steps

1. Run with example data to understand flow
2. Try different knob settings
3. Add your own data
4. Customize for your task
5. Share your results!

Happy experimenting! 🚀
