# Federated Random Labeling (FedRLGP) for MNIST

SEEDS=(0 1 2 3 4 5) 

DATASET="mnist"  
UNLEARN_EPOCHS=80
CUDA_GAP=3

LR=0.02
LOCAL_EPOCH=5 

# create log directory if it doesn't exist
mkdir -p "logs/${DATASET}"

echo "Running FedRL and FedRLGP on $DATASET (lr=$LR, unlearn_epochs=$UNLEARN_EPOCHS, local_epochs=$LOCAL_EPOCH)"

for SEED in "${SEEDS[@]}"; do
    python -u unlearning.py --seed $SEED --config "config/${DATASET}.yml" --lr $LR --ft_lr 0.001 --unlearn_method FedRL --communicator EFFACE --compressor TopK --k 1 --unlearn_epochs $UNLEARN_EPOCHS --refine_epochs 20 --local_epochs $LOCAL_EPOCH --pretrain_model "FedAvg_${DATASET}_${SEED}" --save "FedRL_${DATASET}_${SEED}" --cuda $(( $SEED % 3 )) --unlearn_acc_threshold 0.08 > "logs/${DATASET}/fedrl_EFFACE_${DATASET}_${SEED}.log" 2>&1 & 

    if (( ($SEED + 1) % $CUDA_GAP == 0 )); then
        wait
    fi
done
wait