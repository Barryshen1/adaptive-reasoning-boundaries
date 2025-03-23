# Adaptive Reasoning Boundaries

This repository contains the implementation for the paper:
> **Adaptive Optimization Methods for Reasoning Boundaries: Dynamic Estimation and Multi-Agent Collaboration**

## Introduction

Chain-of-Thought (CoT) reasoning has significantly enhanced the performance of large language models (LLMs) on complex reasoning tasks. The Reasoning Boundary Framework (RBF) quantifies the upper limits of LLMs' reasoning capabilities. This work introduces three novel approaches to optimize reasoning within these boundaries:

1. **Advanced Minimum Acceptable Reasoning Paths (A-MARP)**: Extends the original MARP with adaptive step calibration and contextual decomposition.
2. **Dynamic Boundary Estimation (DBE)**: A real-time method to probe and adapt to a model's reasoning capabilities during interaction.
3. **Multi-Agent Reasoning Collaboration (MARC)**: A framework that leverages specialized agents with complementary reasoning strengths.

## Installation

```bash
git clone https://github.com/Barryshen1/adaptive-reasoning-boundaries.git
cd adaptive-reasoning-boundaries
pip install -r requirements.txt
