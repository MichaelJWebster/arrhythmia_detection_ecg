# arrhythmia_detection_ecg

A reproduction of Kachuee et al. (2018), *ECG Heartbeat Classification: A Deep Transferable Representation* ([arXiv:1805.00794](https://arxiv.org/abs/1805.00794); `1805.00794v2.pdf`):

1. **`ecg_classify_beats.ipynb`** trains a 1-D CNN to classify single-lead ECG beats from MIT-BIH into five classes (N, S, V, F, Q).
2. **`transfer_learning.ipynb`** transfers that network to myocardial-infarction (MI) detection on the PTB Diagnostic database, and compares frozen features, fine-tuning and training from scratch.

## Models

Both architectures are in `ecg_models.py`, shared by the two notebooks:

- **`paper`** (default): the residual CNN of the paper's Fig. 2. A convolution, five residual blocks (32 kernels of size 5, skip connections, max-pooling of size 5 / stride 2), then two fully connected layers; about 54k parameters. It trains with the paper's recipe: Adam, with the learning rate multiplied by 0.75 every 10,000 steps, and rare classes oversampled to balance with random augmentation. (The paper does not say how it augmented; amplitude scaling, time stretching and light noise are this repo's choices.)
- **`simple`**: an earlier, non-residual CNN adapted from [this implementation](https://github.com/sajiddeboss/ECG-HeartBeat-Classification), about 668k parameters, trained with AdamW, ReduceLROnPlateau and a class-weighted loss.

## Data

Download the [Kaggle ECG Heartbeat Categorization dataset](https://www.kaggle.com/datasets/shayanfazeli/heartbeat) and unzip it into `./data`, so that `./data/mitbih_train.csv`, `./data/mitbih_test.csv`, `./data/ptbdb_normal.csv` and `./data/ptbdb_abnormal.csv` exist.

The CSVs hold individual beats with no patient IDs, so train/test splits are by beat, not by patient: beats from the same patient can appear on both sides, which makes scores optimistic for unseen patients.

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch            # pick the variant for your hardware: https://pytorch.org/get-started/locally/
pip install -r requirements.txt
jupyter notebook ecg_classify_beats.ipynb
```

Run `ecg_classify_beats.ipynb` first; `transfer_learning.ipynb` starts from its best checkpoint for the same architecture. Training uses `accelerator="auto"`: CUDA on an NVIDIA GPU, MPS on Apple Silicon, otherwise CPU.

## Settings (environment variables)

| Variable | Default | Effect |
|---|---|---|
| `ECG_DATA_DIR` | `./data` | Where the CSVs are |
| `ECG_ARCH` | `paper` | `paper` or `simple`; used by both notebooks |
| `ECG_BALANCE` | `augment` for `paper`, `weights` for `simple` | Class imbalance: balanced sampling with augmentation, or a class-weighted loss |
| `ECG_EPOCHS` | 100 | Epoch limit for arrhythmia training (early stopping usually ends it first) |
| `MI_EPOCHS` | per model | Overrides every epoch limit in `transfer_learning.ipynb` |

Set `ECG_EPOCHS=1` or `MI_EPOCHS=1` for a quick smoke test.
