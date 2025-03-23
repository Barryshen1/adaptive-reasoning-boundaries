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

### Dataset Preparation

```bash
# Clone the repository
git clone https://github.com/Barryshen1/adaptive-reasoning-boundaries.git
cd adaptive-reasoning-boundaries

# Install dependencies
pip install -r requirements.txt

# Download datasets
python scripts/download_datasets.py
```

## 🔧 Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## 🚀 Running Experiments

### A-MARP

```bash
python run_amarp.py --dataset GSM8K \
                    --model gpt-4 \
                    --alpha 0.15 \
                    --beta 0.08 \
                    --c_max 5 \
                    --output_dir results/amarp
```

### DBE

```bash
python run_dbe.py --dataset BIGGSM \
                 --model gpt-4 \
                 --gamma 0.12 \
                 --probe_frequency 5 \
                 --probe_size 7 \
                 --output_dir results/dbe
```

### MARC

```bash
python run_marc.py --dataset HotpotQA \
                  --models gpt-4,claude-3.5-sonnet,llama-3.1-70b \
                  --agent_roles planner,calculator,verifier,integrator \
                  --communication_rounds 5 \
                  --output_dir results/marc
```

### Running All Methods

```bash
python run_all.py --dataset GSM8K \
                 --model gpt-4 \
                 --output_dir results/all
```

## 📈 Evaluation

To evaluate the performance of our methods:

```bash
python evaluate.py --results_dir results/amarp \
                   --metrics accuracy,boundary_error,token_efficiency \
                   --visualize
```

For ablation studies:

```bash
python ablation.py --method amarp \
                   --components adaptive_step,difficulty_aware,contextual_boundary,memory_augmented \
                   --dataset BIGGSM \
                   --model gpt-4
```

## 📊 Analysis

### Visualization

Generate performance comparison plots:

```bash
python visualize.py --results_dir results \
                    --plot boundary_performance,cross_model,difficulty_analysis \
                    --output_dir figures
```

### Reports

Generate detailed analysis reports:

```bash
python generate_report.py --results_dir results \
                          --output_file report.md
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


