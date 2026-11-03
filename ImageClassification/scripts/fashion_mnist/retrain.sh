# FedRetrain on Fashion-MNIST

SEEDS=(0 1 2 3 4 5)
DATASETS=("fashion_mnist")

# create log directory if it doesn't exist
mkdir -p logs/fashion_mnist

for DATASET in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        python -u unlearning.py --seed $SEED --config "config/${DATASET}.yml" --unlearn_method FedRetrain --pretrain_model "FedAvg_${DATASET}_${SEED}" --save FedRetrain_${DATASET}_${SEED} --cuda $(($SEED % 3)) > "logs/${DATASET}/FedRetrain_${SEED}.log" 2>&1 &
    done 
done