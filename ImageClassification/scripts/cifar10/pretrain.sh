# FedAvg on CIFAR-10

SEEDS=(0 1 2 3 4 5)
DATASETS=("cifar10")

# create log directory if not exist
mkdir -p logs/pretrain

for DATASET in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do 
        python -u unlearning.py --seed $SEED --config "config/${DATASET}.yml" --save FedAvg_${DATASET}_${SEED} --cuda $(($SEED % 3)) > "logs/pretrain/pretrain_${DATASET}_${SEED}.log" 2>&1 &
    done   
    wait
done