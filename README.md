# ARES: Adaptive Reasoning via Entropy Shaping

## Abstract
Recent advances in multimodal large reasoning models (MLRMs) have improved performance on complex textual and visual tasks, yet models still **overthink** on easy problems (producing long, redundant traces) while **under-exploring** on hard ones. **ARES** is a unified framework for **difficulty-aware adaptive reasoning**:
- **Adaptive Cold-Start (AdaCS).** Curates multimodal & textual data with reasoning lengths proportional to difficulty to instill explicit difficulty awareness.
- **Adaptive-Entropy Policy Optimization (AEPO).** Uses **high-window-entropy (HWE)** tokens as reliable triggers for *when* to branch exploration, and a hierarchical entropy reward with **dynamic KL** as the *how much* thinking budget.
- **Effectiveness & Efficiency.** ARES improves accuracy and reduces unnecessary tokens across MathVerse, MathVision, MathVista, DynaMath, LogicVista, MMMU and text-only benchmarks, with strong gains at both 3B and 7B scales.

---

## 🛠️ Installation
```bash
conda create -n ares python=3.11 -y && conda activate ares
# clone your repo first
cd ARES
pip install -e .
```

If you have connectivity issues with Hugging Face mirrors:
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

---

## 🚀 Training

### 1) Cold Start (AdaCS)
```bash
bash ./cold_start/run_cold_start.sh
```

### 2) Staged RL: AEPO
```bash
# example script to prepare rewards / launch AEPO
bash ./examples/run_aepo.sh
```

**Key ideas.**
- **HWE trigger:** branch only in sustained-uncertainty regions.
- **Difficulty-aware shaping:** suppress over-exploration on easy, encourage deeper exploration on hard, stabilize around a batch target on medium.
- **Dynamic KL:** token-wise KL budget that relaxes inside validated HWE windows.

---

## 🧩 Checkpoint Merge (HF format)
```bash
python scripts/model_merger.py \
  --local_dir checkpoints/${ProjectName}/exp_name/global_step_1/actor
```

---

## 📈 Evaluation

Unified evaluation for multimodal & textual reasoning datasets.

**Usage**
```plain
usage: main.py [-h] --model-name MODEL_NAME --openai-api-key OPENAI_API_KEY
               [--openai-base-url OPENAI_BASE_URL] [--cache-dir CACHE_DIR]
               [--output-dir OUTPUT_DIR] [--max-tokens MAX_TOKENS]
               [--min-pixels MIN_PIXELS] [--max-pixels MAX_PIXELS]
               [--temperature TEMPERATURE] [--top-p TOP_P]
               [--system-prompt SYSTEM_PROMPT]
               [--datasets DATASETS] [--dataset-dir DATASET_DIR]
               [--eval-threads EVAL_THREADS] [--max-retries MAX_RETRIES]
```

**Example 1: OpenAI API**
```bash
python ./src/main.py \
  --model-name "gpt-4.1" \
  --openai-api-key "YOUR_API_KEY" \
  --cache-dir "./cache" \
  --datasets "mathverse,mathvision,mathvista,dynamath,mmmu"
```

**Example 2: Local model via lmdeploy**
```bash
lmdeploy serve api_server \
  /path/to/local/lmm \
  --model-name lmm_name \
  --server-port 23333 \
  --chat-template qwen2d5-vl

python ./src/main.py \
  --model-name "lmm_name" \
  --openai-base-url "http://localhost:23333/v1" \
  --openai-api-key "DUMMY_KEY" \
  --cache-dir "./cache" \
  --datasets "all"
```

---

## 📊 Highlights
- **ARES-3B:** +8.4 average over prior open 3B models across core multimodal benchmarks.
- **ARES-7B:** +9.7 average over strong 7B open baselines; large gains on MathVision and DynaMath.
- **Efficiency:** Shorter responses on easy/medium tasks; deeper but targeted exploration on hard tasks.

(See paper tables/figures for complete numbers and ablations on entropy reward vs. dynamic KL.)

---

## 📁 Suggested Repo Structure
```
ARES/
├─ cold_start/                 # AdaCS curation & SFT launchers
├─ aepo/                       # AEPO training loop, rewards, KL scheduler
├─ src/                        # evaluation entrypoints & dataset adapters
├─ scripts/                    # utilities (merger, converters, logging)
├─ examples/                   # runnable scripts (run_aepo.sh, etc.)
└─ README.md
```

---

## 🔍 FAQ
**Q: Why window entropy (HWE) rather than single-token entropy?**  
A: Single-token entropy is noisy (punctuation/formula artifacts). HWE captures *sustained* uncertainty—better aligned with genuine reasoning bifurcations—so we branch only where it matters.

**Q: How does dynamic KL help?**  
A: It serves as a token-wise “thinking budget,” relaxing at validated HWE windows while keeping regularization tight elsewhere—improving both stability and accuracy.

---

## 📜 Citation
```bibtex
@article{ARES_2026,
  title   = {ARES: Multimodal Adaptive Reasoning via Difficulty-Aware Token-Level Entropy Shaping},
  author  = {Anonymous},
  journal = {Under review at ICLR},
  year    = {2026}
}
```

---

## 📎 License
Apache-2.0 (unless otherwise noted in subdirectories).

---

## 🙌 Acknowledgements
We thank the open-source community for tools, datasets, and prior work on reasoning-oriented pretraining and RL that inspired this project.
