SEEDS=(0 1 2 3 4 5)
DATASETS=("cifar100")

for SEED in "${SEEDS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do 
        python -u unlearning.py --seed $SEED --config "config/${DATASET}.yml" --save FedAvg_${DATASET}_${SEED} --cuda $(($SEED % 3)) > "logs/pretrain/pretrain_${DATASET}_${SEED}.log" 2>&1 &
    done   
done