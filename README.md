# ARES

## Abstract

Recent advances in multimodal large reasoning models (MLRMs) have substantially improved their ability to solve complex textual and visual tasks. However, these models still exhibit a fundamental inefficiency: they tend to **overthink** on simple problems, producing unnecessarily lengthy reasoning traces, while **under-exploring** on challenging ones, leading to missed solutions.

We introduce **ARES** (Adaptive Reasoning via Entropy Shaping), a unified open-source framework for **difficulty-aware adaptive reasoning**. Our key contributions are:

1. **Adaptive Cold-Start (AdaCS):** we curate multimodal and textual datasets with reasoning traces of length proportional to problem difficulty, equipping models with explicit difficulty awareness.  
2. **Adaptive Entropy Policy Optimization (AEPO):** we propose a reinforcement learning algorithm that leverages high window-entropy tokens as exploration triggers to decide *when to explore*, while dynamic KL control and hierarchical entropy rewards determine *how much to explore*.  
3. Extensive experiments across diverse mathematical, logical, and multimodal benchmarks demonstrate that ARES achieves **state-of-the-art performance** and **reasoning efficiency**, closing the gap to leading commercial systems while requiring significantly lower inference costs.

---

## Usage

### (Step 1) Install

    conda create -n ares python=3.11 -y && conda activate ares
    git clone https://github.com/xxx/ARES.git
    cd ARES
    pip install -e .

### (Step 2) Training

Cold Start Training (AdaCS)

    bash ./cold_start/run_cold_start.sh

Reinforcement Learning with AEPO

    bash ./rl/run_aepo.sh

If you encounter issues with connecting to Hugging Face, consider:

    export HF_ENDPOINT=https://hf-mirror.com

### (Step 3) Merge Checkpoint in Hugging Face Format

    python3 scripts/model_merger.py --local_dir checkpoints/${PROJECT}/exp_name/global_step_1/actor
