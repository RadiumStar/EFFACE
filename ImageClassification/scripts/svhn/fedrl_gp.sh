# Federated Random Labeling with Gradient Projection (FedRL-GP) for SVHN

SEEDS=(0 1 2 3 4 5) 
 
DATASET="svhn"
UNLEARN_EPOCHS=80
LOCAL_EPOCHS=1
LR=0.005
CUDA_GAP=6

# create log directory if it doesn't exist
mkdir -p "logs/${DATASET}"

# run the unlearning process with FedRL-GP method
echo "Running FedRL-GP on $DATASET (lr=$LR, unlearn_epochs=$UNLEARN_EPOCHS, local_epochs=$LOCAL_EPOCHS)" 

for SEED in "${SEEDS[@]}"; do
    python -u unlearning.py --seed $SEED --config "config/${DATASET}.yml" --lr $LR --unlearn_method FedRLGP --cos_threshold 0 --unlearn_epochs $UNLEARN_EPOCHS --local_epochs $LOCAL_EPOCHS --refine_epochs 0 --pretrain_model "FedAvg_${DATASET}_${SEED}" --save "FedRL_GP_${DATASET}_${SEED}" --cuda $(($SEED % 3)) > "logs/${DATASET}/fedrlgp_${DATASET}_${SEED}.log" 2>&1 &

    if (( ($SEED + 1) % $CUDA_GAP == 0 )); then
        wait
    fi
done  