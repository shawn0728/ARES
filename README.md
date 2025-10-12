<div align="center">



  <h1 style="margin: 0; font-size: 1.8em;">
    <img src="./figures/logo.png" alt="Revisual Icon" width="50" style="vertical-align: middle; margin-right: 10px;">
    ARES: Multimodal Adaptive Reasoning via Difficulty-Aware Token-Level Entropy Shaping
  </h1>

  [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2510.08457)
  [![alphaXiv](https://img.shields.io/badge/discussion-A42C25?style=for-the-badge&logo=arxiv&logoColor=white&color=blue)](https://www.alphaxiv.org/abs/2510.08457)
  [![Github](https://img.shields.io/badge/ARES-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/shawn0728/ARES)
  [![Hugging Face Collection](https://img.shields.io/badge/ARES_Collection-fcd022?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/collections/ares0728/ares-68e7c7160dcb48734dee4e95)

  [![Awesome](https://awesome.re/badge.svg)](https://github.com/shawn0728/ARES)
  [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
  ![](https://img.shields.io/github/last-commit/shawn0728/ARES?color=green) 

</div>

## Abstract
Recent advances in multimodal large reasoning models (MLRMs) have substantially
improved their ability to solve complex textual and visual tasks. However, these
models tend to *overthink* on
simple problems, producing unnecessarily lengthy reasoning traces, while
*under-exploring* on challenging ones, leading to missed solutions. To 
address this imbalance, we propose **ARES**, a unified open-source framework
for *adaptive reasoning* that dynamically allocates exploration effort based
on task difficulty. Our approach is motivated by two key empirical findings:
(i) while single-token entropy is noisy, *high window-entropy (HWE)
tokens* (token-level entropies averaged under a sliding window) can reliably capture reasoning-critical moments; and (ii) reducing HWE usage
benefits easy problems, while increasing it is essential for solving hard ones.
Building on these insights, ARES introduces a two-stage training pipeline. In the
*Adaptive Cold-Start* stage, we curate multimodal and textual data paired
with reasoning traces of length proportional to problem difficulty, equipping the
model with initial difficulty awareness. In the second stage, we develop
*Adaptive Entropy Policy Optimization (AEPO)*, which uses HWE tokens as
exploration triggers to decide *when to explore*, and a hierarchical entropy
reward with dynamic KL control to decide *how much to explore*. Extensive
experiments demonstrate that ARES achieves superior performance and
reasoning efficiency across diverse mathematical, logical, and multimodal
benchmarks, while closing the gap to leading commercial systems under
significantly lower inference costs. 


![Flow chart](assets/final_flowchart_fixed.png)
---

## 🛠️ Installation
```bash
conda create -n aepo python=3.11 -y
conda activate aepo
pip install -r requirements.txt
```

---

## 🚀 Training

### Staged RL: AEPO
```bash
# example script to prepare rewards / launch AEPO
bash ./experiments/AEPO/train.sh
```

**Key ideas.**
- **HWE trigger:** branch only in sustained-uncertainty regions.
- **Difficulty-aware shaping:** suppress over-exploration on easy, encourage deeper exploration on hard, stabilize around a batch target on medium.
- **Dynamic KL:** token-wise KL budget that relaxes inside validated HWE windows.

---

## 🧩 Checkpoint Merge (HF format)
```bash
python scripts/model_merger.py \
  --local_dir ./checkpoints/${ProjectName}/exp_name/global_step_1/actor
```

---

## 📊 Highlights
- **ARES-3B:** +8.4 average over prior open 3B models across core multimodal benchmarks.
- **ARES-7B:** +9.7 average over strong 7B open baselines; large gains on MathVision and DynaMath-W.
- **Efficiency:** Shorter responses on easy/medium tasks; deeper but targeted exploration on hard tasks.

(See paper tables/figures for complete numbers and ablations on entropy reward vs. dynamic KL.)

---

## 🔍 FAQ
**Q: Why window entropy (HWE) rather than single-token entropy?**  
A: Single-token entropy is noisy (punctuation/formula artifacts). HWE captures *sustained* uncertainty—better aligned with genuine reasoning bifurcations—so we branch only where it matters.

**Q: How does dynamic KL help?**  
A: It serves as a token-wise “thinking budget,” relaxing at validated HWE windows while keeping regularization tight elsewhere—improving both stability and accuracy.


---

## 🙌 Acknowledgements
We thank the open-source community for tools, datasets, and prior work on reasoning-oriented pretraining and RL that inspired this project.
