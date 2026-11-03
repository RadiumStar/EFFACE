SEEDS=(0 1 2 3 4 5) 

DATASETS=("mnist")  
UNLEARN_EPOCHS_ARR=(100)
LOCAL_EPOCHS_ARR=(1)
CUDA_GAPS=(3)
LRS_ARR=(0.01)

# with Uncompressed communication
for i in "${!DATASETS[@]}"; do
    DATASET="${DATASETS[$i]}"
    UNLEARN_EPOCHS="${UNLEARN_EPOCHS_ARR[$i]}"
    LOCAL_EPOCHS="${LOCAL_EPOCHS_ARR[$i]}"
    LR="${LRS_ARR[$i]}"
    CUDA_GAP="${CUDA_GAPS[$i]}"

    # create log directory if it doesn't exist
    mkdir -p "logs/${DATASET}"

    # run the unlearning process with FedRL method
    echo "Running FedRL on $DATASET (lr=$LR, unlearn_epochs=$UNLEARN_EPOCHS, local_epochs=$LOCAL_EPOCHS)" 
    
    for SEED in "${SEEDS[@]}"; do
        python -u unlearning.py --seed $SEED --config "config/${DATASET}.yml" --lr $LR --unlearn_method FedRLGP --cos_threshold 0 --unlearn_epochs $UNLEARN_EPOCHS --local_epochs $LOCAL_EPOCHS --refine_epochs 0 --pretrain_model "FedAvg_${DATASET}_${SEED}" --save "FedRL_${DATASET}_${SEED}" --cuda 3 --unlearn_acc_threshold 0.05 > "logs/${DATASET}/fedrlgp_${DATASET}_${SEED}.log" 2>&1 &

        if (( ($SEED + 1) % $CUDA_GAP == 0 )); then
            wait
        fi
    done 
    wait
done  

#####

# Federated Random Labeling (FedRLGP) for MNIST

SEEDS=(0 1 2 3 4 5) 

DATASETS=("mnist")  
UNLEARN_EPOCHS_ARR=(120)
LOCAL_EPOCHS_ARR=(1)
CUDA_GAPS=(3)

LRS_ARR=(0.018)  # Updated learning rates for MNIST 

# with EFFACE communicator and TopK compressor (k=1)
for i in "${!DATASETS[@]}"; do
    DATASET="${DATASETS[$i]}"
    UNLEARN_EPOCHS="${UNLEARN_EPOCHS_ARR[$i]}"
    LOCAL_EPOCHS="${LOCAL_EPOCHS_ARR[$i]}"
    LR="${LRS_ARR[$i]}"
    CUDA_GAP="${CUDA_GAPS[$i]}"
    
    # create log directory if it doesn't exist
    mkdir -p "logs/${DATASET}"

    # run the unlearning process with FedRL method
    echo "Running FedRL on $DATASET (lr=$LR, unlearn_epochs=$UNLEARN_EPOCHS, local_epochs=$LOCAL_EPOCHS)" 
    
    for LR in "${LRS_ARR[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            python -u unlearning.py --seed $SEED --config "config/${DATASET}.yml" --lr $LR --ft_lr 0.0008 --momentum 0.9 --unlearn_method FedRL --communicator EFFACE --compressor TopK --k 1 --unlearn_epochs $UNLEARN_EPOCHS --refine_epochs 50 --local_epochs $LOCAL_EPOCHS --pretrain_model "FedAvg_${DATASET}_${SEED}" --save "FedRL_${DATASET}_${SEED}" --cuda 3 --unlearn_acc_threshold 0.08 > "logs/${DATASET}/fedrl_EFFACE_${DATASET}_${SEED}.log" 2>&1 &

            python -u unlearning.py --seed $SEED --config "config/${DATASET}.yml" --lr $LR --ft_lr 0.0008 --momentum 0.9 --unlearn_method FedRLGP --cos_threshold 0 --communicator EFFACE --compressor TopK --k 1 --unlearn_epochs $UNLEARN_EPOCHS --refine_epochs 50 --local_epochs $LOCAL_EPOCHS --pretrain_model "FedAvg_${DATASET}_${SEED}" --save "FedRL_${DATASET}_${SEED}" --cuda 3 --unlearn_acc_threshold 0.08 > "logs/${DATASET}/fedrlgp_EFFACE_${DATASET}_${SEED}.log" 2>&1 &

            if (( ($SEED + 1) % $CUDA_GAP == 0 )); then
                wait
            fi
        done 
        wait
    done
done