# FedRetrain on CIFAR-100

SEEDS=(0 1 2 3 4 5)
DATASETS=("cifar100")

# echo "Running FedRetrain on $DATASET"
for DATASET in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        python -u unlearning.py --seed $SEED --config "config/${DATASET}.yml" --lr 0.01 --unlearn_method FedRetrain --save FedRetrain_${DATASET}_${SEED} --cuda $(($SEED % 3)) > "logs/${DATASET}/FedRetrain_${SEED}.log" 2>&1 &
        if (( ($SEED + 1) % 3 == 0 )); then
            wait
        fi
    done
done 
