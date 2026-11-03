# seeds 
SEEDS=(0 1 2 3 4)
ALPHAS=(0.1 0.5 1.0 1e8)

for ALPHA in ${ALPHAS[@]}
do
    for SEED in ${SEEDS[@]}
    do
        echo "Pretraining with seed ${SEED} and alpha ${ALPHA}..."
        python -u pretrain.py --seed ${SEED} --alpha ${ALPHA} --global_epochs 10000 > logs/pretrain_seed${SEED}_alpha${ALPHA}.log 2>&1
    done
done