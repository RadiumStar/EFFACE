# Federated Unlearning with Gradient Ascent for MNIST

SEEDS=(0 1 2 3 4 5)
DATASET="mnist"
UNLEARN_EPOCHS_ARR=(30 30 30 30 20 30)
REFINE_EPOCHS_ARR=(30 30 30 30 80 30)
LR=0.012
FT_LRS=(0.001 0.001 0.001 0.001 0.001 0.001)
WEIGHT_DECAY=0.05

for SEED in "${SEEDS[@]}"; do 
    # create log directory if it doesn't exist
    mkdir -p "logs/${DATASET}"

    # run the unlearning process with FedGA method
    echo "Running FedGA on $DATASET (unlearn_epochs=${UNLEARN_EPOCHS_ARR[$SEED]}, refine_epochs=${REFINE_EPOCHS_ARR[$SEED]})" 
    
    python -u unlearning.py --seed $SEED --config "config/${DATASET}.yml" --lr $LR --ft_lr ${FT_LRS[$SEED]} --weight_decay $WEIGHT_DECAY --unlearn_method FedGA --unlearn_epochs ${UNLEARN_EPOCHS_ARR[$SEED]} --refine_epochs ${REFINE_EPOCHS_ARR[$SEED]} --pretrain_model "FedAvg_${DATASET}_${SEED}" --save "FedGA_${DATASET}_${SEED}" --unlearn_acc_threshold 0.05 --cuda $(($SEED % 3)) > "logs/${DATASET}/fedga_${DATASET}_${SEED}.log" 2>&1 & 
done

wait