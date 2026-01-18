# EFFACE Code Overview 

```
EFFACE/
├── code/
│   ├── config/               # configuration files
│   ├── core/                 # core federated unlearning algorithms implementation
│   ├── datasets/             # dataset handling and preprocessing
│   ├── models/               # Model architectures 
│   ├── saves/                # Saved models, checkpoints & client data indices
│   ├── utils/                # Utility functions including compressors & communicators (Error Feedback)
│   ├── init.py               # Initialization file
│   └── unlearning.py         # Main script for federated unlearning
```

## Requirements

- Python 3.9+

```
pip install -r requirements.txt
```

## Scripts

1. Prepare your data and place it in the `../../data/` directory or replace the path in the `datasets/your_dataset.py` file.
2. Run the pretraining program firstly, for instance, cifar-100 here:

    ```bash
    python unlearning.py --config config/cifar100.yml --save FedAvg_cifar100
    ```

    You may get your pretrained model in `saves/models/FedAvg_cifar100.pth`

3. Run the unlearning program, for instance, cifar-100 here:

    ```bash
    python unlearning.py --config config/cifar100.yml --communicator EFFACE --compressor TopK --k 1 --lr 0.005 --unlearn_method FedRL --pretrain_model FedAvg_cifar100 --unlearn_epochs 30 --refine_epochs 0
    ```

## Core Algorithms Implementation

The core federated unlearning algorithms are implemented in the `core/` directory, including:

- fedavg: just for pretraining
- fedretrain: retraining from scratch
- fedgd: federated gradient descent/fine-tuning unlearning
- fedga: federated gradient ascent unlearning
- fedrl: federated random-label based unlearning, which is also the ***basic unlearning method*** with EFFACE

While other implemented baselines can be found in the table below:

| Baseline  | Description                          | Implementation | Note |
|-----------|--------------------------------------|-----|-----|
| [FedGB](https://ieeexplore.ieee.org/document/11228643/)     | federated gradient balancing/gradient difference unlearning | [FedUOSC](https://github.com/RadiumStar/FedUOSC) | The version in [FedUOSC](https://ieeexplore.ieee.org/document/11228643/) without compression |
| [FedUOSC](https://ieeexplore.ieee.org/document/11228643/) | federated unlearning with oriented saliency compression | [FedUOSC](https://github.com/RadiumStar/FedUOSC) | We consider the default implementation in [FedUOSC](https://ieeexplore.ieee.org/document/11228643/) with 50% compression ratio |
| [FedPSGA](https://arxiv.org/abs/2207.05521) | federated projected stochastic gradient ascent unlearning | [FedPSGA](https://github.com/IBM/federated-unlearning) | Note that the original implementation only provided an unlearning method for client removal. Here we regard the unlearning dataset as a special client and do federated projected stochastic gradient ascent on it, followed by post training on the remaining clients |
| [NoT](https://arxiv.org/abs/2503.05657) | federated unlearning via weight negation | Section 11 of [NoT](https://arxiv.org/abs/2503.05657) | We consider the default implementation in [NoT](https://arxiv.org/abs/2503.05657) with only negating the first layer of all models |
