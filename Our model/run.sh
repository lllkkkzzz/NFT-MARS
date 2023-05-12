
for COLLECTION in azuki bayc coolcats doodles meebits
do
for NUM_EPOCH in 50
do
for SEED in 2023
do

    python main.py \
        --collection $COLLECTION \
        --num_epoch $NUM_EPOCH \
        --seed $SEED &

done
done
done