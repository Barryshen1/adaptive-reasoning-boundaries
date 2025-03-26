<p align="center">
<h1 align="center">Adaptive Optimization Methods for Reasoning Boundaries: Dynamic Estimation and Multi-Agent Collaboration</h1>
</p>

<p align="center">
  	<a href="https://img.shields.io/badge/version-v0.1.0-blue">
      <img alt="version" src="https://img.shields.io/badge/version-v0.1.0-blue?color=FF8000?color=009922" />
    </a>
    <a >
       <img alt="PRs-Welcome" src="https://img.shields.io/badge/PRs-Welcome-blue" />
  	</a>
    <br />
</p>

## 🔍 Overview

This repository contains the implementation of three novel approaches to optimize reasoning within model boundaries:

1. **Advanced Minimum Acceptable Reasoning Paths (A-MARP)**: Extends the original MARP with adaptive step calibration, difficulty-aware decomposition, and contextual boundary awareness.

2. **Dynamic Boundary Estimation (DBE)**: A real-time method that probes model capabilities during interaction, adaptively estimating reasoning boundaries and adjusting prompting strategies accordingly.

3. **Multi-Agent Reasoning Collaboration (MARC)**: A framework leveraging specialized agents with complementary reasoning strengths, implementing task-specific delegation and consensus mechanisms.

## 📊 Datasets

We evaluate our methods on seven reasoning datasets across multiple domains:

- **BIGGSM**: Extended version of the dataset containing complex arithmetic problems with varying calculation amounts.
- **GSM8K**: Grade school math word problems requiring multi-step reasoning.
- **MATH**: Advanced mathematics problems across various topics.
- **MultiArith**: Elementary arithmetic word problems.
- **HotpotQA**: Multi-hop question answering requiring reasoning across documents.
- **StrategyQA**: Questions requiring implicit multi-step reasoning and strategy.
- **MGSM**: Multilingual mathematical reasoning dataset.

## Dataset Preparation

The project uses seven reasoning datasets across multiple domains. You can prepare these datasets using the following steps:

### Automatic Dataset Downloading

Most datasets will be automatically downloaded when you run experiments for the first time. The `data/loaders/dataset_loaders.py` script handles dataset downloading via the Hugging Face datasets library.

### Manual Dataset Preparation (if needed)

For datasets that require manual preparation:

1. **BIGGSM**: If not available through automatic download, create a directory:
   ```bash
   mkdir -p data/biggsm
   ```
   Place your BIGGSM dataset file in this directory as `data.jsonl`.

2. **Custom Datasets**: To use your own datasets, add them to the appropriate directories under `data/` and implement a loader in `data/loaders/dataset_loaders.py`.

### Dataset Configuration

You can configure dataset parameters in your experiment commands:
```bash
python run_experiments.py --dataset gsm8k --sample_size 50 --difficulty_control True
```

Parameters:
- `--dataset`: Name of the dataset (gsm8k, math, biggsm, hotpotqa, strategyqa, multiarith, mgsm)
- `--sample_size`: Number of examples to sample
- `--difficulty_control`: Whether to create controlled difficulty test sets

## 🔧 Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## 🚀 Running Experiments

Set up your API keys (for OpenAI or Anthropic)

```bash
# For OpenAI
export OPENAI_API_KEY=your_api_key_here

# For Anthropic
export ANTHROPIC_API_KEY=your_api_key_here
```

### Run experiments with different methods

```bash
# Run A-MARP on GSM8K with GPT-4
python run_experiments.py --method a_marp --dataset gsm8k --model gpt-4 --sample_size 50

# Run DBE on GSM8K with GPT-4
python run_experiments.py --method dbe --dataset gsm8k --model gpt-4 --sample_size 50

# Run MARC on GSM8K with GPT-4
python run_experiments.py --method marc --dataset gsm8k --model gpt-4 --sample_size 50

# Run standard CoT baseline on GSM8K with GPT-4
python run_experiments.py --method standard_cot --dataset gsm8k --model gpt-4 --sample_size 50
```

## 📈 Evaluation

Evaluate results:

```bash
# Evaluate a specific method
python evaluation/evaluate.py --method A-MARP

# Compare all methods
python evaluation/evaluate.py --method all --dataset gsm8k
```

Analyze and visualize results:

```bash
# Generate visualizations
python analyze_results.py --visualize
```

## 🧪 Experimental Results

Our methods consistently outperform baseline approaches across all datasets:

- A-MARP provides a XX% average improvement over the original MARP.
- DBE demonstrates adaptive capability with up to XX% improvement after 10 interactions.
- MARC shows the largest improvement, with a XX% gain on BIGGSM and XX% on HotpotQA compared to standard CoT.

Key strengths of our approaches:

- A-MARP is particularly effective in the Partially Feasible Reasoning Boundary (PFRB) regions.
- DBE's boundary estimation significantly improves with more interactions, demonstrating its learning capability.
- MARC leverages complementary reasoning strengths, with cross-model teams showing the best performance.

## 🧩 Project Structure

```
adaptive-reasoning-boundaries/
├── README.md
├── requirements.txt
├── methods/
│   ├── __init__.py
│   ├── a_marp.py
│   ├── dbe.py
│   └── marc.py
├── data/
│   ├── __init__.py
│   └── loaders/
│       ├── __init__.py
│       └── dataset_loaders.py
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py
│   └── evaluate.py
├── experiments/
│   ├── __init__.py
│   ├── configs/
│   │   ├── a_marp_config.py
│   │   ├── dbe_config.py
│   │   └── marc_config.py
│   └── results/
├── utils/
│   ├── __init__.py
│   ├── data.py
│   ├── tools.py
│   └── request_tool.py
├── run_experiments.py
└── analyze_results.py
```

## 📚 Citation

If you find our work useful, please consider citing:

```bibtex
@inproceedings{author2025adaptive,
    title = "Adaptive Optimization Methods for Reasoning Boundaries: Dynamic Estimation and Multi-Agent Collaboration",
    author = "Author, Anonymous",
    booktitle = "Proc. of NeurIPS",
    year = "2025",
}
```

## 🔗 Related Work

Our work builds directly on the Reasoning Boundary Framework (RBF):

```bibtex
@inproceedings{chen-etal-2024-rg,
    title = "Unlocking the Boundaries of Thought: A Reasoning Granularity Framework to Quantify and Optimize Chain-of-Thought",
    author = "Chen, Qiguang  and
      Qin, Libo  and
      Jiaqi, Wang  and
      Jinxuan, Zhou  and
      Che, Wanxiang",
    booktitle = "Proc. of NeurIPS",
    year = "2024",
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact

For questions or feedback, please create GitHub issues or contact the authors.

---

## Acknowledgments
