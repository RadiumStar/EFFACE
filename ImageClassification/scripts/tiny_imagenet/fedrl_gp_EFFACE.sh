# Federated Random Labeling with Gradient Projection (FedRL-GP) for Tiny ImageNet

SEEDS=(0 1 2 3 4 5)
DATASETS=("tiny_imagenet")

for DATASET in "${DATASETS[@]}"; do
    # create log directory if it doesn't exist
    mkdir -p "logs/${DATASET}"

    # run the unlearning process with FedRLGP method
    echo "Running FedRLGP on $DATASET, set cos_threshold to 0"
    for SEED in "${SEEDS[@]}"; do
        python -u unlearning.py --seed $SEED --config "config/${DATASET}.yml" --lr 0.005 --unlearn_method FedRLGP --cos_threshold 0 --k 1 --communicator EFFACE --compressor TopK --unlearn_epochs 100 --local_epochs 2 --refine_epochs 0 --pretrain_model FedAvg_${DATASET}_${SEED} --save FedRLGP_${DATASET}_${SEED} --cuda $(($SEED % 3)) > "logs/${DATASET}/fedrlgp_${DATASET}_${SEED}.log" 2>&1 &

        if (( ($SEED + 1) % 3 == 0 )); then
            wait
        fi
    done    
done