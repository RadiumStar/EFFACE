# Ablation Study: EFFACE with different etas

mkdir -p "logs/ablation_etas"

SEEDS=(0 1 2 3 4 5)
COMMUNICATOR="EFFACE"
DATASETS=("cifar10" "svhn" "cifar100" "tiny_imagenet")
UNLEARN_EPOCHS_ARR=(80 80 100 100)
LOCAL_EPOCHS_ARR=(1 1 1 2)
LRS_ARR=(0.001 0.005 0.005 0.005)
ETAS_ARR=(0.1 0.05 0.01 0.005 0.001)

CUDA_GAPS=(6 6 3 3)

for i in "${!DATASETS[@]}"; do
    DATASET="${DATASETS[$i]}"
    UNLEARN_EPOCHS="${UNLEARN_EPOCHS_ARR[$i]}"
    LOCAL_EPOCHS="${LOCAL_EPOCHS_ARR[$i]}"
    LR="${LRS_ARR[$i]}"
    CUDA_GAP="${CUDA_GAPS[$i]}"

    # run the unlearning process with different etas
    for ETA in "${ETAS_ARR[@]}"; do
        echo "Running FedRL on $DATASET with communicator $COMMUNICATOR with eta $ETA" 
        
        for SEED in "${SEEDS[@]}"; do
            echo "Running FedRL on $DATASET with communicator $COMMUNICATOR and eta $ETA (lr=$LR, unlearn_epochs=$UNLEARN_EPOCHS, local_epochs=$LOCAL_EPOCHS) for seed $SEED"
            python -u unlearning.py --seed $SEED --config "config/${DATASET}.yml" --communicator $COMMUNICATOR --eta $ETA --compressor TopK --k 1 --lr $LR --unlearn_method FedRL --unlearn_epochs $UNLEARN_EPOCHS --local_epochs $LOCAL_EPOCHS --refine_epochs 0 --pretrain_model "FedAvg_${DATASET}_${SEED}" --save "FedRL_${DATASET}_${SEED}" --cuda $(($SEED % 3)) > "logs/ablation_etas/fedrl_${ETA}_${DATASET}_${SEED}.log" 2>&1 &

            if (( ($SEED + 1) % $CUDA_GAP == 0 )); then
                wait
            fi
        done 
        wait

        echo "Running FedRLGP on $DATASET with communicator $COMMUNICATOR with eta $ETA"
        for SEED in "${SEEDS[@]}"; do
            echo "Running FedRLGP on $DATASET with communicator $COMMUNICATOR and eta $ETA (lr=$LR, unlearn_epochs=$UNLEARN_EPOCHS, local_epochs=$LOCAL_EPOCHS) for seed $SEED"
            python -u unlearning.py --seed $SEED --config "config/${DATASET}.yml" --communicator $COMMUNICATOR --eta $ETA --compressor TopK --k 1 --lr $LR --unlearn_method FedRLGP --cos_threshold 0 --unlearn_epochs $UNLEARN_EPOCHS --local_epochs $LOCAL_EPOCHS --refine_epochs 0 --pretrain_model "FedAvg_${DATASET}_${SEED}" --save "FedRLGP_${DATASET}_${SEED}" --cuda $(($SEED % 3)) > "logs/ablation_etas/fedrlgp_${ETA}_${DATASET}_${SEED}.log" 2>&1 &

            if (( ($SEED + 1) % $CUDA_GAP == 0 )); then
                wait
            fi
        done
        wait
    done
done