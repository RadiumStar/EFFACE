# Federated Unlearning with Projected Gradient Ascent for Fashion-MNIST

SEEDS=(0 1 2 3 4 5)
DATASET="fashion_mnist"
UNLEARN_EPOCHS_ARR=(7 8 7 7 8 7)
REFINE_EPOCHS_ARR=(100 100 100 100 100 100)
LR=0.01

SEEDS=(1)

for SEED in "${SEEDS[@]}"; do 
    # create log directory if it doesn't exist
    mkdir -p "logs/${DATASET}"

    UNLEARN_EPOCHS=${UNLEARN_EPOCHS_ARR[$SEED]}
    REFINE_EPOCHS=${REFINE_EPOCHS_ARR[$SEED]}
    # run the unlearning process with FedPSGA method
    echo "Running FedPSGA on $DATASET (unlearn_epochs=${UNLEARN_EPOCHS}, refine_epochs=${REFINE_EPOCHS})" 
    
    python -u unlearning.py --seed $SEED --config "config/${DATASET}.yml" --lr $LR --ft_lr 0.008 --unlearn_method FedPSGA --unlearn_epochs ${UNLEARN_EPOCHS} --refine_epochs ${REFINE_EPOCHS} --local_epochs 2 --pretrain_model "FedAvg_${DATASET}_${SEED}" --save "FedPSGA_${DATASET}_${SEED}" --unlearn_acc_threshold 0.09 --cuda $(($SEED % 3)) > "logs/${DATASET}/fedpsga_${DATASET}_${SEED}.log" 2>&1 & 
done

wait