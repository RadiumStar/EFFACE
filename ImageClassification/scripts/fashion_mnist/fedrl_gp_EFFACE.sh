# Federated Random Labeling (FedRLGP) for Fashion-MNIST

SEEDS=(0 1 2 3 4 5) 

DATASET="fashion_mnist" 
UNLEARN_EPOCHS=80
CUDA_GAP=6

LR=0.01
LOCAL_EPOCH=5

# create log directory if it doesn't exist
mkdir -p "logs/${DATASET}"

# with EFFACE communicator and TopK compressor (k=1) 
# run the unlearning process with FedRL method
echo "Running FedRL(GP) on $DATASET (unlearn_epochs=$UNLEARN_EPOCHS, local_epochs=$LOCAL_EPOCH, lr=$LR)" 

for SEED in "${SEEDS[@]}"; do 
    python -u unlearning.py --seed $SEED --config "config/${DATASET}.yml" --lr $LR --ft_lr 0.001 --unlearn_method FedRLGP --cos_threshold 0 --communicator EFFACE --compressor TopK --k 1 --unlearn_epochs $UNLEARN_EPOCHS --local_epochs $LOCAL_EPOCH --refine_epochs 20 --pretrain_model "FedAvg_${DATASET}_${SEED}" --save "FedRL_${DATASET}_${SEED}" --cuda $(( $SEED % 3 )) --unlearn_acc_threshold 0.08 >> "logs/${DATASET}/fedrlgp_EFFACE_${DATASET}_${SEED}.log" 2>&1 &

    if (( ($SEED + 1) % $CUDA_GAP == 0 )); then
        wait
    fi
done 
wait 