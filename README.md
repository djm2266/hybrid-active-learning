# Hybrid Active Learning with Counterfactuals and Human Oracle

A flexible active learning system that intelligently routes samples between counterfactual generation and human annotation based on uncertainty, novelty, coverage, and feasibility metrics.

## Overview

This implementation follows Algorithm 1 from the paper, providing a **tunable knob** to control the balance between:
- **Counterfactual Generation**: Automated label generation via LLM-based text transformations
- **Human Oracle**: Gold-standard human annotations

### Key Features

- **Dynamic Routing**: Intelligent sample selection based on multiple criteria
- **Multi-Metric Scoring**: Uncertainty, novelty, coverage, and CF-feasibility
- **Adaptive Thresholds**: Percentile-based cutoffs that evolve each round
- **Risk Assessment**: Safety scoring to flag potentially problematic samples
- **Weighted Training**: Lower weights for CF-generated vs. human-labeled data
- **Resume Capability**: Checkpoint support for long-running experiments
- **Multiple LLM Providers**: Support for Ollama, Gemini, OpenAI

## Algorithm Flow

```
1. Score Samples: u(x), d(x), c(x), g(x), r(x)
2. Compute Dynamic Thresholds: τ_h, τ_l, δ, γ, c*
3. Route for Labeling:
   - High uncertainty + (high coverage need OR high novelty) → Human
   - Medium uncertainty + high CF-feasibility + low risk → Counterfactual
   - Otherwise → Defer to next round
4. Obtain Labels:
   - Human oracle: weight = 1.0
   - Counterfactual: weight = 0.7 (configurable)
5. Update Model with weighted samples
6. Adapt thresholds for next round
```

## Installation

### Prerequisites

- Python 3.8+
- Virtual environment recommended

### Setup

```bash
# Clone repository
git clone <repo-url>
cd hybrid-active-learning

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify setup
python check_setup.py
```

### LLM Provider Setup

#### Option 1: Ollama (Local, Free)
```bash
# Install Ollama from https://ollama.ai
ollama serve  # Start server
ollama pull qwen2.5:7b  # Pull model
```

#### Option 2: Gemini (Cloud, API Key Required)
```bash
# Get API key from https://makersuite.google.com/app/apikey
# Add to config.yaml:
llm:
  provider: gemini
  gemini:
    api_key: YOUR_API_KEY_HERE
    model: gemini-1.5-flash
```

#### Option 3: OpenAI (Cloud, API Key Required)
```bash
# Add to config.yaml:
llm:
  provider: openai
  openai:
    api_key: YOUR_API_KEY_HERE
    model: gpt-4o-mini
```

## Quick Start

### 1. Prepare Your Data

Place training data in `data/input/`:

```csv
id,text,label
1,"Sample text here",positive
2,"Another example",negative
```

Required columns: `id`, `text`, `label`

### 2. Configure Settings

Edit `config/config.yaml`:

```yaml
active_learning:
  knob:
    cf_weight: 0.5  # 0.0 = all human, 1.0 = all CF
    human_budget: 100  # Max human annotations per round
    cf_budget: 200     # Max CF generations per round
  
  thresholds:
    uncertainty_high_percentile: 80
    uncertainty_low_percentile: 40
    novelty_percentile: 70
    feasibility_percentile: 60
    coverage_percentile: 70
```

### 3. Run Pipeline

```bash
# Full pipeline
python run_pipeline.py

# Or step-by-step
python src/01_data_formatting.py
python src/02_counterfactual_over_generation.py
python src/03_counterfactual_filtering.py
python src/04_counterfactual_evaluation.py
python src/05_active_learning_loop.py
```

## Configuration Options

### The Knob: Balancing CF vs Human

The `cf_weight` parameter (0.0 to 1.0) controls routing:

- **cf_weight = 0.0**: Pure human oracle (traditional active learning)
- **cf_weight = 0.3**: Mostly human, some CF for easy cases
- **cf_weight = 0.5**: Balanced hybrid (default)
- **cf_weight = 0.7**: Mostly CF, human for hard cases
- **cf_weight = 1.0**: Pure counterfactual generation

### Threshold Tuning

```yaml
thresholds:
  # High uncertainty → likely needs human
  uncertainty_high_percentile: 80  # Top 20% uncertainty
  
  # Low uncertainty → possibly suitable for CF
  uncertainty_low_percentile: 40   # Bottom 60% excluded
  
  # Novel samples → prefer human to avoid CF errors
  novelty_percentile: 70           # Top 30% novelty
  
  # CF-feasible samples
  feasibility_percentile: 60       # Top 40% feasibility
  
  # Coverage need
  coverage_percentile: 70          # Top 30% need coverage
```

### Budget Constraints

```yaml
active_learning:
  budget:
    total_rounds: 10
    samples_per_round: 100
    human_budget: 50       # Human annotations per round
    cf_budget: 100         # CF generations per round
    cost_human: 1.0        # Relative cost
    cost_cf: 0.1           # CF is cheaper
```

## Metrics and Scoring

### 1. Uncertainty Score (u)
```
u(x) = -Σ p_k log(p_k) / log(K)
```
Normalized entropy over K classes.

### 2. Novelty Score (d)
```
d(x) = min_{z_i ∈ L} ||z_x - z_i||_2
```
Distance to nearest labeled example in embedding space.

### 3. Coverage Score (c)
```
c(x) = 1 - |C_j| / max_i |C_i|
```
Inverse cluster size (prioritize underrepresented clusters).

### 4. CF-Feasibility Score (g)
```
g(x) = 0.5(1 - tanh(0.05(len(x) - L_0))) + 0.5 * max_i a_i / Σ a_i
```
Based on text length and attention weights.

### 5. Risk Score (r)
```
r(x) = P_unsafe(x) + α(1 - g(x))
```
Safety score + infeasibility penalty.

## Routing Logic

```python
if u(x) >= τ_h and (c(x) >= c* or d(x) >= δ) or r(x) is high:
    → Send to Human Oracle
    
elif τ_l <= u(x) < τ_h and g(x) >= γ and r(x) is low:
    → Send to Counterfactual Generation
    
else:
    → Defer to Next Round
```

## Output Files

```
data/output/
├── [seed][model]_candidate_phrases_annotated_data.csv
├── [seed][model]_counterfactuals_train.csv
├── [seed][model]_filtered_counterfactuals_train.csv
├── al_round_{n}_human_annotations.csv
├── al_round_{n}_cf_generated.csv
├── al_round_{n}_model_checkpoint.pt
└── al_final_results.json
```

## Monitoring and Evaluation

Each round produces:
- **Global F1**: Overall model performance
- **Per-cluster metrics**: F1 for each cluster/class
- **Boundary health**: Decision boundary quality
- **CF acceptance rate**: Percentage of valid counterfactuals
- **Cost tracking**: Human vs CF annotation costs
- **Coverage gaps**: Underrepresented regions

## Advanced Usage

### Custom Scoring Functions

```python
# In src/active_learning/scorer.py
def custom_uncertainty(model, x, embeddings):
    # Your custom uncertainty metric
    return score

# Register in config
scoring:
  uncertainty_fn: custom_uncertainty
```

### Human-in-the-Loop Integration

```python
from src.active_learning.oracle import HumanOracle

oracle = HumanOracle(interface='web')  # or 'cli', 'api'
label = oracle.get_label(sample)
```

### Experiment Tracking

```python
# Enable MLflow tracking
tracking:
  enabled: true
  backend: mlflow
  experiment_name: hybrid_al_experiment
```

## Troubleshooting

### Common Issues

1. **API Quota Exhausted**
   - Script automatically saves checkpoints
   - Resume with: `python src/02_counterfactual_over_generation.py`
   - Checkpoints stored in `data/output/interim/`

2. **Low CF Acceptance Rate**
   - Increase `feasibility_percentile` (more selective)
   - Adjust `filtering.similarity_threshold` in config
   - Use better LLM model

3. **Imbalanced Routing**
   - Tune `cf_weight` knob
   - Adjust threshold percentiles
   - Check budget constraints

4. **Memory Issues**
   - Reduce `samples_per_round`
   - Use smaller embedding model
   - Enable batch processing

## Project Structure

```
hybrid-active-learning/
├── config/
│   ├── config.yaml              # Main configuration
│   └── prompts/                 # LLM prompt templates
├── src/
│   ├── active_learning/
│   │   ├── scorer.py           # Metric computation
│   │   ├── router.py           # Sample routing logic
│   │   ├── oracle.py           # Human oracle interface
│   │   └── trainer.py          # Weighted model training
│   ├── counterfactual/
│   │   ├── generator.py        # CF generation
│   │   ├── validator.py        # CF validation
│   │   └── filtering.py        # Quality filtering
│   ├── utils/
│   │   ├── llm_providers.py    # LLM API wrappers
│   │   ├── embeddings.py       # Text embeddings
│   │   └── metrics.py          # Evaluation metrics
│   ├── 01_data_formatting.py
│   ├── 02_counterfactual_over_generation.py
│   ├── 03_counterfactual_filtering.py
│   ├── 04_counterfactual_evaluation.py
│   └── 05_active_learning_loop.py
├── data/
│   ├── input/                   # Training data
│   └── output/                  # Results
├── notebooks/                   # Analysis notebooks
├── tests/                       # Unit tests
├── check_setup.py              # Dependency checker
├── run_pipeline.py             # Main runner
├── requirements.txt
└── README.md
```

## Citation

If you use this code, please cite:

```bibtex
@article{hybrid_active_learning,
  title={Hybrid Active Learning with Counterfactuals and Human Oracle},
  author={Your Name},
  year={2024}
}
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## Support

- Issues: [GitHub Issues](https://github.com/yourusername/hybrid-active-learning/issues)
- Docs: [Full Documentation](https://your-docs-url.com)
- Contact: your.email@example.com
