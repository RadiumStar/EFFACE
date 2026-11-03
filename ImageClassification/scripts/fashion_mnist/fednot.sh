# Federated Unlearning via Weight Negation for Fashion-MNIST

SEEDS=(0 1 2 3 4 5)
DATASETS=("fashion_mnist")

UNLEARN_EPOCHS_ARR=(120)
LOCAL_EPOCHS_ARR=(1)
LRS_ARR=(0.001)
CUDA_GAPS=(6)

for i in "${!DATASETS[@]}"; do  
    DATASET="${DATASETS[$i]}"
    UNLEARN_EPOCHS="${UNLEARN_EPOCHS_ARR[$i]}"
    LOCAL_EPOCHS="${LOCAL_EPOCHS_ARR[$i]}"
    LR="${LRS_ARR[$i]}"
    CUDA_GAP="${CUDA_GAPS[$i]}"
    
    # create log directory if it doesn't exist
    mkdir -p "logs/${DATASET}"

    # run the unlearning process with FedNot method
    echo "Running FedNot on $DATASET (global_epochs=$UNLEARN_EPOCHS, local_epochs=$LOCAL_EPOCHS)" 
    
    for SEED in "${SEEDS[@]}"; do
        python -u unlearning.py --seed $SEED --config "config/${DATASET}.yml" --lr $LR --unlearn_method FedNot --global_epochs $UNLEARN_EPOCHS --local_epochs $LOCAL_EPOCHS --pretrain_model "FedAvg_${DATASET}_${SEED}" --save "FedNot_${DATASET}_${SEED}" --cuda $(($SEED % 3)) > "logs/${DATASET}/fednot_${DATASET}_${SEED}.log" 2>&1 &

        if (( ($SEED + 1) % $CUDA_GAP == 0 )); then
            wait
        fi
    done 
    wait
done