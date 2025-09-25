# ARES: Adaptive Reasoning via Entropy Shaping

## Abstract
Recent advances in multimodal large reasoning models (MLRMs) have improved performance on complex textual and visual tasks, yet models still **overthink** on easy problems (producing long, redundant traces) while **under-exploring** on hard ones. **ARES** is a unified framework for **difficulty-aware adaptive reasoning**:
- **Adaptive Cold-Start (AdaCS).** Curates multimodal & textual data with reasoning lengths proportional to difficulty to instill explicit difficulty awareness.
- **Adaptive-Entropy Policy Optimization (AEPO).** Uses **high-window-entropy (HWE)** tokens as reliable triggers for *when* to branch exploration, and a hierarchical entropy reward with **dynamic KL** as the *how much* thinking budget.
- **Effectiveness & Efficiency.** ARES improves accuracy and reduces unnecessary tokens across MathVerse, MathVision, MathVista, DynaMath, LogicVista, MMMU and text-only benchmarks, with strong gains at both 3B and 7B scales.


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
