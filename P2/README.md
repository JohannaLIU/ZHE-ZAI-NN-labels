PART Ⅱ
# ZHE Stage II: neural training (silver → gold)

This repository contains a single script `nn_training_(1).py`, a neural training pipeline for the **ZHE state classifier** in **Stage II** of the project  
(Stage I = corpus collection/splitting + CI/TM/CA; Stage II = neural training).

## What it does
- Train on **silver-labeled** ZHE data
- Evaluate / calibrate on a **gold** subset (supports silver-only, gold-only, or hybrid runs)
- Two settings:
  - **M0**: context-only
  - **M1**: context + layer-feature inputs
- Model: **CharCNN–BiLSTM–Attention**

## Requirements
Python 3.9+ recommended.
- numpy, pandas, torch, scikit-learn, matplotlib, adjustText

## Data layout (expected)
Before running, prepare the dataset files in the layout required by the path variables inside the script.
(See the config/path block at the top of `nn_training_(1).py`.)

>
>- The script will export a gold sample file ('zhe_gold_batch01_final_annot_with_qc.csv' in the block '=== 10% double-check sampling + marked column ===') that requires manual review.
- After review, save it as a reviewed version, which will then be used as input for subsequent gold training.
- This evaluation is also universal for silver labels' credibility initially produced in 'P1'.
