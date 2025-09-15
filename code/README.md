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

4. For other unlearning baselines, you can change the `--unlearn_method` arguments