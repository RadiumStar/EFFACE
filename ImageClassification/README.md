# Federated Unlearning for Image Classification

This folder contains the image-classification experiments of *Top-1 Compression Suffices for Federated Unlearning with the Help of Adaptive Error Feedback* (ICASSP 2026).

All experiments share a single entry point, `unlearning.py`, which performs

- **FedAvg pretraining** (`--unlearn_method FedAvg`),
- **federated unlearning** — our methods (`FedRL`, `FedRLGP`) and all baselines


## Repository Layout

```
ImageClassification/
├── config/                  # one YAML per dataset (mnist, fashion_mnist, svhn, cifar10, cifar100, tiny_imagenet)
├── core/                    # federated (un)learning algorithms as client/server pairs
├── datasets/                # dataset wrappers, client partitioning, backdoor injection, (un)learning index selection
├── models/                  # model zoo (MLP / CNN / ResNet variants)
├── scripts/                 # experiment launchers: one folder per dataset, plus `ablation/`
│   ├── ablation/            # communicator / compressor / eta ablations
│   ├── cifar10/  cifar100/  svhn/  mnist/  fashion_mnist/  tiny_imagenet/
├── saves/                   # generated at runtime: models, checkpoints, client data indices
├── utils/                   # communicators & compressors (error feedback) and misc helpers
├── init.py                  # argument parser (YAML defaults + CLI overrides)
├── unlearning.py            # main entry point
└── requirements.txt
```

## Requirements

- Python 3.9+
- (optional) CUDA-capable GPU

```bash
pip install -r requirements.txt
```

The pinned versions (`torch==1.13.1`, `torchvision==0.14.1`, `numpy==1.26.4`, `Pillow`, `pyyaml`) reproduce our environment; newer versions are generally compatible as well.

## Quick Start: CIFAR-100

All commands below are executed **inside this `ImageClassification/` directory**, because every relative path (`config/`, `saves/`, `logs/`, `../../data/`) is resolved from here.

### Step 0 — Data

The dataset modules read from `../../data/`, e.g. `datasets/cifar100.py` uses `../../data/cifar100/`. Torchvision-based datasets (`mnist`, `fashion_mnist`, `svhn`, `cifar10`, `cifar100`) are downloaded automatically on the first run; Tiny-ImageNet has to be placed manually under `../../data/tiny-imagenet-200/`.

### Step 1 — FedAvg pretraining

```bash
mkdir -p logs/pretrain
bash scripts/cifar100/pretrain.sh          # seeds 0-5
```

which is equivalent to running one seed at a time:

```bash
python -u unlearning.py --config config/cifar100.yml \
    --seed 0 --save FedAvg_cifar100_0 --cuda 0
```

The best global model is written to `saves/models/FedAvg_cifar100_0.pth`.

### Step 2 — Unlearning with EFFACE

```bash
bash scripts/cifar100/fedrl_EFFACE.sh      # FedRL + EFFACE + Top-1, seeds 0-5
```

which expands to:

```bash
python -u unlearning.py --seed 0 --config config/cifar100.yml \
    --unlearn_method FedRL \
    --communicator EFFACE --compressor TopK --k 1 \
    --lr 0.005 --unlearn_epochs 100 --local_epochs 1 --refine_epochs 0 \
    --pretrain_model FedAvg_cifar100_0 --save FedRL_cifar100_0 --cuda 0
```

producing

- model: `saves/models/FedRL_cifar100_0.pth`
- log: `logs/cifar100/fedrl_EFFACE_cifar100_0.log`

You can also try the other unlearning methods / datasets in `scripts/`, or override any config key on the command line. 

## Core Algorithms Implementation

The core federated unlearning algorithms are implemented in the `core/` directory, including:


| Baseline  | Description                          | Implementation | Note |
|-----------|--------------------------------------|-----|-----|
| FedRetrain | retraining from scratch after removing unlearned samples | - | - |
| [FedGD](https://ieeexplore.ieee.org/document/11228665) | fine-tuning on the remaining dataset from the orignal model | - | - |
| [FedGA](https://ieeexplore.ieee.org/abstract/document/9964015) | federatd gradient ascent on the forgetting dataset | - | - |
| [FedPSGA](https://arxiv.org/abs/2207.05521) | federated projected stochastic gradient ascent unlearning | [FedPSGA](https://github.com/IBM/federated-unlearning) | Note that the original implementation only provided an unlearning method for client removal. Here we regard the unlearning dataset as a special client and do federated projected stochastic gradient ascent on it, followed by post training on the remaining clients |
| FedGB [[1](https://arxiv.org/abs/2504.05822), [2](https://ieeexplore.ieee.org/document/11228643/)]     | federated gradient balancing/gradient difference unlearning | [FedUOSC](https://github.com/RadiumStar/FedUOSC) | The version in [FedUOSC](https://ieeexplore.ieee.org/document/11228643/) without compression |
| [FedUOSC](https://ieeexplore.ieee.org/document/11228643/) | federated unlearning with oriented saliency compression | [FedUOSC](https://github.com/RadiumStar/FedUOSC) | We consider the default implementation in [FedUOSC](https://ieeexplore.ieee.org/document/11228643/) with 50% compression ratio |
| [NoT](https://arxiv.org/abs/2503.05657) | federated unlearning via weight negation | Section 11 of [NoT](https://arxiv.org/abs/2503.05657) | We consider the default implementation in [NoT](https://arxiv.org/abs/2503.05657) with only negating the first layer of all models |
| [FedMemPrune](https://arxiv.org/abs/2605.24545) | federated unlearning via memory pruning | [FedMemPrune](https://github.com/JWei1999/Rethinking-Federated-Unlearning-via-the-Lens-of-Memorization) | We consider the pruning ratio in line 94 - 105 in [core/fedmemprune.py](core/fedmemprune.py)|
| [FedRL](https://ieeexplore.ieee.org/document/10825563) | federated unlearning with random labeling | [SalUn](https://github.com/OPTML-Group/Unlearn-Saliency/) | We consider the random labeling loss in [SalUn](https://arxiv.org/abs/2306.17192) as the default implementation of FedRL |

## Communicators and Compressors

`utils/communicators.py` and `utils/compressors.py` implement the transmission layer. They are selected purely by name in `get_communicator_class` (`utils/__init__.py`), so a new method can be combined with any of them without touching `core/`.

| Communicator | Description |
|---|---|
| `Communicator` | no error feedback (uncompressed baseline) |
| [EF14](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/IS140694.pdf) | error feedback with gradient compression and compensated error |
| [EF21](https://arxiv.org/abs/2106.05203) | error feedback with gradient difference compression |
| [EF21+](https://arxiv.org/abs/2106.05203) | EF21+: variant of EF21 |
| [EControl](https://proceedings.iclr.cc/paper_files/paper/2024/file/bce52456a36be2be1abd95427139de37-Paper-Conference.pdf) | error feedback with gradient difference compression and compensated error scaled by `--eta` |
| EFFACE (Ours) | adaptive error feedback with the selection between gradient compression and gradient difference compression, and compensated error scaled by `--eta` |

| Compressor | Description |
|---|---|
| `Compressor` | identity, no compression |
| `TopK` | keep the `k` largest-magnitude coordinates (`--k`) or a `--sparsity` fraction |
| `RandK` | keep `k` random coordinates |
| `SignSGD` | sign quantization |
| `QSGD` | stochastic quantization (`--bits`) |
| `TernGrad` | ternary quantization | 


## Notes

- Delete `saves/indices/*.pkl` (or change `seed` / `unlearn_*` / `partition` / `alpha`) to re-sample client partitions and unlearned subsets.
- All launchers assume they are executed from `ImageClassification/`; running them from elsewhere breaks the relative `config/`, `saves/` and `logs/` paths.
- Adding a dataset requires `config/<name>.yml`, `datasets/<name>.py`, and a `scripts/<name>/` folder following the same pattern.
