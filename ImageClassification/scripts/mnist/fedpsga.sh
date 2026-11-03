# Federated Unlearning with Projected Gradient Ascent for MNIST

SEEDS=(0 1 2 3 4 5)
DATASET="mnist"
UNLEARN_EPOCHS_ARR=(25 32 13 19 14 28)
REFINE_EPOCHS_ARR=(80 80 80 80 100 100)
LR=0.02
FT_LRS=(0.01 0.01 0.01 0.01 0.01 0.005) 

SEEDS=(4 5)

for SEED in "${SEEDS[@]}"; do 
    # create log directory if it doesn't exist
    mkdir -p "logs/${DATASET}"

    UNLEARN_EPOCHS=${UNLEARN_EPOCHS_ARR[$SEED]}
    REFINE_EPOCHS=${REFINE_EPOCHS_ARR[$SEED]}
    REFINE_EPOCHS=$(($REFINE_EPOCHS - $UNLEARN_EPOCHS))
    FT_LR=${FT_LRS[$SEED]}

    # run the unlearning process with FedPSGA method
    echo "Running FedPSGA on $DATASET (unlearn_epochs=${UNLEARN_EPOCHS_ARR[$SEED]}, refine_epochs=${REFINE_EPOCHS_ARR[$SEED]})" 
    
    python -u unlearning.py --seed $SEED --config "config/${DATASET}.yml" --lr $LR --ft_lr $FT_LR --unlearn_method FedPSGA --unlearn_epochs $UNLEARN_EPOCHS --refine_epochs $REFINE_EPOCHS --pretrain_model "FedAvg_${DATASET}_${SEED}" --save "FedPSGA_${DATASET}_${SEED}" --unlearn_acc_threshold 0.1 --cuda $(($SEED % 3)) > "logs/${DATASET}/fedpsga_${DATASET}_${SEED}.log" 2>&1 & 
done

wait