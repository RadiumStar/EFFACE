# Federated Random Labeling with Gradient Projection (FedRL-GP) for SVHN

DATASET="svhn"
LR=0.005

# create log directory if it doesn't exist
mkdir -p "logs/${DATASET}"

# run the unlearning process with FedRLGP method
echo "Running FedRLGP on $DATASET, set cos_threshold to 0"
for SEED in "${SEEDS[@]}"; do
    python -u unlearning.py --seed $SEED --config "config/${DATASET}.yml" --lr $LR --unlearn_method FedRLGP --cos_threshold 0 --communicator EFFACE --compressor TopK --k 1 --unlearn_epochs 80 --refine_epochs 0 --pretrain_model FedAvg_${DATASET}_${SEED} --save FedRLGP_${DATASET}_${SEED} --cuda $(($SEED % 3)) > "logs/${DATASET}/fedrlgp_${DATASET}_${SEED}.log" 2>&1 &

    if (( ($SEED + 1) % 3 == 0 )); then
        wait
    fi 
done