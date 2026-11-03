# Federated Unlearning with Oriented Saliency Compression for MNIST

SEEDS=(0 1 2 3 4 5)
DATASET="mnist"
UNLEARN_EPOCHS=15
REFINE_EPOCHS=65
LRS=(0.08 0.08 0.08 0.08 0.03 0.08)
LOCAL_EPOCHS=(2 2 2 2 4 5) 
THRESHOLDS=(0.01 0.05 0.01 0.05 0.05 0.05)
FT_LR=0.01

SEEDS=(0 2)
for SEED in "${SEEDS[@]}"; do
    LR=${LRS[$SEED]}
    LOCAL_EPOCH=${LOCAL_EPOCHS[$SEED]}
    THRESHOLD=${THRESHOLDS[$SEED]}

    echo "Running FedUOSC on $DATASET (unlearn_epochs=$UNLEARN_EPOCHS, refine_epochs=$REFINE_EPOCHS, lr=$LR, local_epochs=$LOCAL_EPOCH)" 
    # create log directory if it doesn't exist
    mkdir -p "logs/${DATASET}"

    # run the unlearning process with FedUOSC method
    python -u unlearning.py --seed $SEED --config "config/${DATASET}.yml" --lr $LR --ft_lr $FT_LR --unlearn_method FedUOSC --unlearn_epochs $UNLEARN_EPOCHS --refine_epochs $REFINE_EPOCHS --local_epochs $LOCAL_EPOCH --pretrain_model "FedAvg_${DATASET}_${SEED}" --save "FedUOSC_${DATASET}_${SEED}" --unlearn_acc_threshold $THRESHOLD --cuda $(( $SEED % 3 )) > "logs/${DATASET}/feduosc_${DATASET}_${SEED}.log" 2>&1 & 
done 
wait 