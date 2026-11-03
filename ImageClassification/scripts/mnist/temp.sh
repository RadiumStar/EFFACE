SEEDS=(0) 

DATASETS=("mnist")  
UNLEARN_EPOCHS_ARR=(100)
LOCAL_EPOCHS_ARR=(1)
CUDA_GAPS=(4)

LRS_ARR=(0.01 0.015 0.018 0.02)
MOMENTUM_ARR=(0.95 0.99)

# clean the log files first 
# echo "" > "logs/mnist/temp_mnist_0.log" 

# with EFFACE communicator and TopK compressor (k=1)
for i in "${!DATASETS[@]}"; do
    DATASET="${DATASETS[$i]}"
    UNLEARN_EPOCHS="${UNLEARN_EPOCHS_ARR[$i]}"
    LOCAL_EPOCHS="${LOCAL_EPOCHS_ARR[$i]}"
    CUDA_GAP="${CUDA_GAPS[$i]}"
    SEED="${SEEDS[$i]}"
    
    # create log directory if it doesn't exist
    mkdir -p "logs/${DATASET}"

    # run the unlearning process with FedRL method
    
    for MOMENTUM in "${MOMENTUM_ARR[@]}"; do
        for LR in "${LRS_ARR[@]}"; do
            echo "Running FedRL on $DATASET (lr=$LR, momentum=$MOMENTUM, unlearn_epochs=$UNLEARN_EPOCHS)" 

            python -u unlearning.py --seed $SEED --config "config/${DATASET}.yml" --lr $LR --ft_lr 0.001 --momentum $MOMENTUM --unlearn_method FedRL --communicator EFFACE --compressor TopK --k 1 --unlearn_epochs $UNLEARN_EPOCHS --refine_epochs 20 --local_epochs $LOCAL_EPOCHS --pretrain_model "FedAvg_${DATASET}_${SEED}" --save "FedRL_${DATASET}_${SEED}" --cuda 3 --unlearn_acc_threshold 0.08 >> "logs/${DATASET}/temp_${DATASET}_${SEED}.log" 2>&1 & 

            wait
        done 
        wait
    done
done
wait