SEEDS=(0 1 2 3 4)
ALPHAS=(0.1 0.5 1.0 1e8)

for ALPHA in ${ALPHAS[@]}
do
    for SEED in ${SEEDS[@]}
    do
        echo "Unlearning with seed ${SEED} and alpha ${ALPHA}..."

        echo "Uncompressed..."
        python -u  ./unlearn.py --communicator Uncompressed --seed ${SEED} --alpha ${ALPHA} --global_epochs 500 --save_trajectory 1 > logs/Uncompressed_seed${SEED}_alpha${ALPHA}_project.log &

        echo "Compressed..."
        python -u  ./unlearn.py --communicator Compressed --seed ${SEED} --alpha ${ALPHA} --global_epochs 500 --save_trajectory 1 > logs/Compressed_seed${SEED}_alpha${ALPHA}_project.log &

        echo "EF14..."
        python -u ./unlearn.py --communicator EF14 --seed ${SEED} --alpha ${ALPHA} --global_epochs 500 --save_trajectory 1 > logs/EF14_seed${SEED}_alpha${ALPHA}_project.log &

        echo "EF21..."
        python -u  ./unlearn.py --communicator EF21  --seed ${SEED} --alpha ${ALPHA} --global_epochs 500 --save_trajectory 1 > logs/EF21_seed${SEED}_alpha${ALPHA}_project.log & 

        echo "EControl..."
        python -u  ./unlearn.py --communicator EControl --seed ${SEED} --alpha ${ALPHA} --global_epochs 500 --save_trajectory 1 > logs/EControl_seed${SEED}_alpha${ALPHA}_project.log &

        echo "EFFACE..."
        python -u  ./unlearn.py --communicator EFFACE --seed ${SEED} --alpha ${ALPHA} --global_epochs 500 --save_trajectory 1 > logs/EFFACE_seed${SEED}_alpha${ALPHA}_project.log &

        wait
    done
done