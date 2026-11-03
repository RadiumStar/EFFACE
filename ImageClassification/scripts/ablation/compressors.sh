# Ablation Study: Compressors including TopK, RandK, SignSGD, QSGD, TernGrad

mkdir -p "logs/ablation_compressors"

SEEDS=(0 1 2 3 4 5)
COMPRESSORS=("RandK" "QSGD" "SignSGD" "TernGrad")   # Top-1 has been tested in the main experiments, so we only test other compressors here.
DATASETS=("cifar10" "svhn" "cifar100" "tiny_imagenet")
UNLEARN_EPOCHS_ARR=(80 80 100 100)
LOCAL_EPOCHS_ARR=(1 1 1 2)
LRS_ARR=(0.001 0.005 0.005 0.005)
CUDA_GAPS=(6 6 3 3)
COMMUNICATOR="EFFACE"

for i in "${!DATASETS[@]}"; do
    DATASET="${DATASETS[$i]}"
    UNLEARN_EPOCHS="${UNLEARN_EPOCHS_ARR[$i]}"
    LOCAL_EPOCHS="${LOCAL_EPOCHS_ARR[$i]}"
    LR="${LRS_ARR[$i]}"
    CUDA_GAP="${CUDA_GAPS[$i]}"

    # run the unlearning process with different communicators
    for COMPRESSOR in "${COMPRESSORS[@]}"; do
        echo "Running FedRL on $DATASET with compressor $COMPRESSOR (lr=$LR, unlearn_epochs=$UNLEARN_EPOCHS, local_epochs=$LOCAL_EPOCHS)" 
        
        for SEED in "${SEEDS[@]}"; do
            python -u unlearning.py --seed $SEED --config "config/${DATASET}.yml" --communicator $COMMUNICATOR --compressor $COMPRESSOR --k 1 --bits 1 --lr $LR --unlearn_method FedRL --unlearn_epochs $UNLEARN_EPOCHS --local_epochs $LOCAL_EPOCHS --refine_epochs 0 --pretrain_model "FedAvg_${DATASET}_${SEED}" --save "FedRL_${DATASET}_${SEED}_${COMPRESSOR}" --cuda $(($SEED % 3)) > "logs/ablation_compressors/fedrl_${COMPRESSOR}_${DATASET}_${SEED}.log" 2>&1 &

            if (( ($SEED + 1) % $CUDA_GAP == 0 )); then
                wait
            fi
        done 
        wait

        echo "Running FedRL on $DATASET with compressor $COMPRESSOR (lr=$LR, unlearn_epochs=$UNLEARN_EPOCHS, local_epochs=$LOCAL_EPOCHS)" 
        for SEED in "${SEEDS[@]}"; do
            python -u unlearning.py --seed $SEED --config "config/${DATASET}.yml" --communicator $COMMUNICATOR --compressor $COMPRESSOR --k 1 --bits 1 --lr $LR --unlearn_method FedRLGP --cos_threshold 0 --unlearn_epochs $UNLEARN_EPOCHS --local_epochs $LOCAL_EPOCHS --refine_epochs 0 --pretrain_model "FedAvg_${DATASET}_${SEED}" --save "FedRLGP_${DATASET}_${SEED}_${COMPRESSOR}" --cuda $(($SEED % 3)) > "logs/ablation_compressors/fedrlgp_${COMPRESSOR}_${DATASET}_${SEED}.log" 2>&1 &

            if (( ($SEED + 1) % $CUDA_GAP == 0 )); then
                wait
            fi
        done 
        wait
    done
done