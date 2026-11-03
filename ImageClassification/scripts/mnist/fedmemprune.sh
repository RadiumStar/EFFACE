# Federated Memory Pruning (FedMemPrune) for CIFAR-10$(($SEED % 3))

# SEEDS=(0 1 2 3 4 5)
# DATASETS=("cifar10" "svhn" "mnist" "fashion_mnist" "cifar100" "tiny_imagenet")
# GLOBAL_EPOCHS_ARR=(50 50 40 40 80 80)
# CUDA_GAPS=(2 2 3 3 1 1)
# LRS=(0.0001 0.0001 0.001 0.005 0.005 0.005)

DATASET="mnist"
GLOBAL_EPOCHS=40
CUDA_GAP=3
LR=0.001

# create log directory if it doesn't exist
mkdir -p "logs/${DATASET}"

for SEED in "${SEEDS[@]}"; do   
    python -u unlearning.py --seed $SEED --config "config/${DATASET}.yml" --lr $LR --optimizer Adam --global_epochs ${GLOBAL_EPOCHS} --local_epochs 2 --unlearn_method FedMemPrune --pretrain_model "FedAvg_${DATASET}_${SEED}" --save "FedMemPrune_${DATASET}_${SEED}" --unlearn_acc_threshold 0.1 --cuda $(( $SEED % 3 )) > "logs/${DATASET}/fedmemprune_${DATASET}_${SEED}.log" 2>&1 &

    if (( ($SEED + 1) % $CUDA_GAP == 0 )); then
        wait
    fi 
done 