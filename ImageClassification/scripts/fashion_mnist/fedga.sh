# Federated Unlearning with Gradient Ascent for Fashion-MNIST

SEEDS=(0 1 2 3 4 5)
DATASET="fashion_mnist"
UNLEARN_EPOCHS_ARR=(15 15 15 15 15 15)
REFINE_EPOCHS_ARR=(15 15 15 15 15 15)
LR=0.01

for SEED in "${SEEDS[@]}"; do 
    # create log directory if it doesn't exist
    mkdir -p "logs/${DATASET}"

    # run the unlearning process with FedGA method
    echo "Running FedGA on $DATASET (unlearn_epochs=${UNLEARN_EPOCHS_ARR[$SEED]}, refine_epochs=${REFINE_EPOCHS_ARR[$SEED]})" 
    
    python -u unlearning.py --seed $SEED --config "config/${DATASET}.yml" --lr $LR --ft_lr 0.001 --unlearn_method FedGA --unlearn_epochs ${UNLEARN_EPOCHS_ARR[$SEED]} --refine_epochs ${REFINE_EPOCHS_ARR[$SEED]} --pretrain_model "FedAvg_${DATASET}_${SEED}" --save "FedGA_${DATASET}_${SEED}" --unlearn_acc_threshold 0.05 --cuda $(($SEED % 3)) > "logs/${DATASET}/fedga_${DATASET}_${SEED}.log" 2>&1 & 
done

wait