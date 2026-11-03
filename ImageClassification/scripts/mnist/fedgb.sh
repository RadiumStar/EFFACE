# Federated Unlearning with Gradient Balancing for MNIST

SEEDS=(0 1 2 3 4 5)
DATASET="mnist"
UNLEARN_EPOCHS=15
REFINE_EPOCHS=65
LRS=(0.08 0.08 0.08 0.08 0.03 0.08)
LOCAL_EPOCHS=(2 2 2 2 4 5)
MOMENTUM=0.9
FT_LR=0.01

for SEED in "${SEEDS[@]}"; do
    LR=${LRS[$SEED]}
    LOCAL_EPOCH=${LOCAL_EPOCHS[$SEED]}

    echo "Running FedGB on $DATASET (unlearn_epochs=$UNLEARN_EPOCHS, refine_epochs=$REFINE_EPOCHS, lr=$LR, local_epochs=$LOCAL_EPOCH)" 
    # create log directory if it doesn't exist
    mkdir -p "logs/${DATASET}"

    # run the unlearning process with FedGB method
    python -u unlearning.py --seed $SEED --config "config/${DATASET}.yml" --lr $LR --momentum $MOMENTUM --ft_lr $FT_LR --unlearn_method FedGB --unlearn_epochs $UNLEARN_EPOCHS --refine_epochs $REFINE_EPOCHS --local_epochs $LOCAL_EPOCH --pretrain_model "FedAvg_${DATASET}_${SEED}" --save "FedGB_${DATASET}_${SEED}" --unlearn_acc_threshold 0.05 --cuda $(( $SEED % 3 )) > "logs/${DATASET}/fedgb_${DATASET}_${SEED}.log" 2>&1 & 
done 
wait 