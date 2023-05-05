# for DATASET in azuki
# do
# for MODEL in AutoInt
# do
# for CONFIG in context
# do
# for SEED in 2022
# do

#     python main.py \
#         --model $MODEL \
#         --dataset $DATASET \
#         --config $CONFIG \
#         --seed $SEED &

# done
# done
# done
# done

# wait

# for DATASET in bayc
# do
# for MODEL in AutoInt
# do
# for CONFIG in context
# do
# for SEED in 2022
# do

#     python main.py \
#         --model $MODEL \
#         --dataset $DATASET \
#         --config $CONFIG \
#         --seed $SEED &

# done
# done
# done
# done

# wait

# for DATASET in coolcats
# do
# for MODEL in AutoInt
# do
# for CONFIG in context
# do
# for SEED in 2023 
# do

#     python main.py \
#         --model $MODEL \
#         --dataset $DATASET \
#         --config $CONFIG \
#         --seed $SEED &

# done
# done
# done
# done

# wait

# for DATASET in doodles
# do
# for MODEL in AutoInt
# do
# for CONFIG in context
# do
# for SEED in 2024
# do

#     python main.py \
#         --model $MODEL \
#         --dataset $DATASET \
#         --config $CONFIG \
#         --seed $SEED &

# done
# done
# done
# done

# wait

# for DATASET in meebits
# do
# for MODEL in AutoInt
# do
# for CONFIG in context
# do
# for SEED in 2023
# do

#     python main.py \
#         --model $MODEL \
#         --dataset $DATASET \
#         --config $CONFIG \
#         --seed $SEED &

# done
# done
# done
# done

# wait



# for DATASET in azuki
# do
# for MODEL in DCN
# do
# for CONFIG in context
# do
# for SEED in 2024
# do

#     python main.py \
#         --model $MODEL \
#         --dataset $DATASET \
#         --config $CONFIG \
#         --seed $SEED &

# done
# done
# done
# done

# wait

# for DATASET in bayc
# do
# for MODEL in DCN
# do
# for CONFIG in context
# do
# for SEED in 2022
# do

#     python main.py \
#         --model $MODEL \
#         --dataset $DATASET \
#         --config $CONFIG \
#         --seed $SEED &

# done
# done
# done
# done

# wait

# for DATASET in coolcats
# do
# for MODEL in DCN
# do
# for CONFIG in context
# do
# for SEED in 2023
# do

#     python main.py \
#         --model $MODEL \
#         --dataset $DATASET \
#         --config $CONFIG \
#         --seed $SEED &

# done
# done
# done
# done

# wait

# for DATASET in doodles
# do
# for MODEL in DCN
# do
# for CONFIG in context
# do
# for SEED in 2024
# do

#     python main.py \
#         --model $MODEL \
#         --dataset $DATASET \
#         --config $CONFIG \
#         --seed $SEED &

# done
# done
# done
# done

# wait

# for DATASET in meebits
# do
# for MODEL in DCN
# do
# for CONFIG in context
# do
# for SEED in 2024
# do

#     python main.py \
#         --model $MODEL \
#         --dataset $DATASET \
#         --config $CONFIG \
#         --seed $SEED &

# done
# done
# done
# done

# wait






# for DATASET in azuki
# do
# for MODEL in WideDeep
# do
# for CONFIG in context
# do
# for SEED in 2022 2023 2024
# do

#     python main.py \
#         --model $MODEL \
#         --dataset $DATASET \
#         --config $CONFIG \
#         --seed $SEED &

# done
# done
# done
# done

# wait

# for DATASET in bayc
# do
# for MODEL in WideDeep
# do
# for CONFIG in context
# do
# for SEED in 2022 2024
# do

#     python main.py \
#         --model $MODEL \
#         --dataset $DATASET \
#         --config $CONFIG \
#         --seed $SEED &

# done
# done
# done
# done

# wait

# for DATASET in coolcats
# do
# for MODEL in WideDeep
# do
# for CONFIG in context
# do
# for SEED in 2022 2023 2024
# do

#     python main.py \
#         --model $MODEL \
#         --dataset $DATASET \
#         --config $CONFIG \
#         --seed $SEED &

# done
# done
# done
# done

# wait

# for DATASET in doodles
# do
# for MODEL in WideDeep
# do
# for CONFIG in context
# do
# for SEED in 2022 2023 2024
# do

#     python main.py \
#         --model $MODEL \
#         --dataset $DATASET \
#         --config $CONFIG \
#         --seed $SEED &

# done
# done
# done
# done

# wait

# for DATASET in meebits
# do
# for MODEL in WideDeep
# do
# for CONFIG in context
# do
# for SEED in 2022 2023 2024
# do

#     python main.py \
#         --model $MODEL \
#         --dataset $DATASET \
#         --config $CONFIG \
#         --seed $SEED &

# done
# done
# done
# done

# wait


# for DATASET in azuki
# do
# for MODEL in DeepFM
# do
# for CONFIG in context
# do
# for SEED in 2023
# do

#     python main.py \
#         --model $MODEL \
#         --dataset $DATASET \
#         --config $CONFIG \
#         --seed $SEED &

# done
# done
# done
# done

# wait

# for DATASET in bayc
# do
# for MODEL in DeepFM
# do
# for CONFIG in context
# do
# for SEED in 2022 2023 2024
# do

#     python main.py \
#         --model $MODEL \
#         --dataset $DATASET \
#         --config $CONFIG \
#         --seed $SEED &

# done
# done
# done
# done

# wait

for DATASET in coolcats
do
for MODEL in DeepFM
do
for CONFIG in context
do
for SEED in 2022
do

    python main.py \
        --model $MODEL \
        --dataset $DATASET \
        --config $CONFIG \
        --seed $SEED &

done
done
done
done

wait

# for DATASET in doodles
# do
# for MODEL in DeepFM
# do
# for CONFIG in context
# do
# for SEED in 2023 2024
# do

#     python main.py \
#         --model $MODEL \
#         --dataset $DATASET \
#         --config $CONFIG \
#         --seed $SEED &

# done
# done
# done
# done

# wait

# for DATASET in meebits
# do
# for MODEL in DeepFM
# do
# for CONFIG in context
# do
# for SEED in 2023 2024
# do

#     python main.py \
#         --model $MODEL \
#         --dataset $DATASET \
#         --config $CONFIG \
#         --seed $SEED &

# done
# done
# done
# done

# wait





for DATASET in azuki
do
for MODEL in FM
do
for CONFIG in context
do
for SEED in 2023
do

    python main.py \
        --model $MODEL \
        --dataset $DATASET \
        --config $CONFIG \
        --seed $SEED &

done
done
done
done

wait

# for DATASET in bayc
# do
# for MODEL in FM
# do
# for CONFIG in context
# do
# for SEED in 2022
# do

#     python main.py \
#         --model $MODEL \
#         --dataset $DATASET \
#         --config $CONFIG \
#         --seed $SEED &

# done
# done
# done
# done

# wait

# for DATASET in coolcats
# do
# for MODEL in FM
# do
# for CONFIG in context
# do
# for SEED in 2023
# do

#     python main.py \
#         --model $MODEL \
#         --dataset $DATASET \
#         --config $CONFIG \
#         --seed $SEED &

# done
# done
# done
# done

# wait

# for DATASET in doodles
# do
# for MODEL in FM
# do
# for CONFIG in context
# do
# for SEED in 2022 2024
# do

#     python main.py \
#         --model $MODEL \
#         --dataset $DATASET \
#         --config $CONFIG \
#         --seed $SEED &

# done
# done
# done
# done

# wait

# for DATASET in meebits
# do
# for MODEL in FM
# do
# for CONFIG in context
# do
# for SEED in 2023
# do

#     python main.py \
#         --model $MODEL \
#         --dataset $DATASET \
#         --config $CONFIG \
#         --seed $SEED &

# done
# done
# done
# done

# wait






# for DATASET in azuki
# do
# for MODEL in LightGCN
# do
# for CONFIG in general
# do
# for SEED in 2022 2023 2024
# do

#     python main.py \
#         --model $MODEL \
#         --dataset $DATASET \
#         --config $CONFIG \
#         --seed $SEED &

# done
# done
# done
# done

# wait

# for DATASET in bayc
# do
# for MODEL in LightGCN
# do
# for CONFIG in general
# do
# for SEED in 2022 2023 2024
# do

#     python main.py \
#         --model $MODEL \
#         --dataset $DATASET \
#         --config $CONFIG \
#         --seed $SEED &

# done
# done
# done
# done

# wait

# for DATASET in coolcats
# do
# for MODEL in LightGCN
# do
# for CONFIG in general
# do
# for SEED in 2022 2023 2024
# do

#     python main.py \
#         --model $MODEL \
#         --dataset $DATASET \
#         --config $CONFIG \
#         --seed $SEED &

# done
# done
# done
# done

# wait

# for DATASET in doodles
# do
# for MODEL in LightGCN
# do
# for CONFIG in general
# do
# for SEED in 2022 2023 2024
# do

#     python main.py \
#         --model $MODEL \
#         --dataset $DATASET \
#         --config $CONFIG \
#         --seed $SEED &

# done
# done
# done
# done

# wait

# for DATASET in meebits
# do
# for MODEL in LightGCN
# do
# for CONFIG in general
# do
# for SEED in 2022 2023 2024
# do

#     python main.py \
#         --model $MODEL \
#         --dataset $DATASET \
#         --config $CONFIG \
#         --seed $SEED &

# done
# done
# done
# done

# wait






# for DATASET in azuki
# do
# for MODEL in NeuMF
# do
# for CONFIG in general
# do
# for SEED in 2024
# do

#     python main.py \
#         --model $MODEL \
#         --dataset $DATASET \
#         --config $CONFIG \
#         --seed $SEED &

# done
# done
# done
# done

# wait

# for DATASET in bayc
# do
# for MODEL in NeuMF
# do
# for CONFIG in general
# do
# for SEED in 2022
# do

#     python main.py \
#         --model $MODEL \
#         --dataset $DATASET \
#         --config $CONFIG \
#         --seed $SEED &

# done
# done
# done
# done

# wait

# for DATASET in coolcats
# do
# for MODEL in NeuMF
# do
# for CONFIG in general
# do
# for SEED in 2022
# do

#     python main.py \
#         --model $MODEL \
#         --dataset $DATASET \
#         --config $CONFIG \
#         --seed $SEED &

# done
# done
# done
# done

# wait

# for DATASET in doodles
# do
# for MODEL in NeuMF
# do
# for CONFIG in general
# do
# for SEED in 2022 2023 2024
# do

#     python main.py \
#         --model $MODEL \
#         --dataset $DATASET \
#         --config $CONFIG \
#         --seed $SEED &

# done
# done
# done
# done

# wait

# for DATASET in meebits
# do
# for MODEL in NeuMF
# do
# for CONFIG in general
# do
# for SEED in 2024
# do

#     python main.py \
#         --model $MODEL \
#         --dataset $DATASET \
#         --config $CONFIG \
#         --seed $SEED &

# done
# done
# done
# done

# wait

# for DATASET in azuki
# do
# for MODEL in DMF
# do
# for CONFIG in general
# do
# for SEED in 2022 2023 2024
# do

#     python main.py \
#         --model $MODEL \
#         --dataset $DATASET \
#         --config $CONFIG \
#         --seed $SEED &

# done
# done
# done
# done

# wait

# for DATASET in bayc
# do
# for MODEL in DMF
# do
# for CONFIG in general
# do
# for SEED in 2022 2023 2024
# do

#     python main.py \
#         --model $MODEL \
#         --dataset $DATASET \
#         --config $CONFIG \
#         --seed $SEED &

# done
# done
# done
# done

# wait

# for DATASET in coolcats
# do
# for MODEL in DMF
# do
# for CONFIG in general
# do
# for SEED in 2022 2023 2024
# do

#     python main.py \
#         --model $MODEL \
#         --dataset $DATASET \
#         --config $CONFIG \
#         --seed $SEED &

# done
# done
# done
# done

# wait

# for DATASET in doodles
# do
# for MODEL in DMF
# do
# for CONFIG in general
# do
# for SEED in 2022 2023 2024
# do

#     python main.py \
#         --model $MODEL \
#         --dataset $DATASET \
#         --config $CONFIG \
#         --seed $SEED &

# done
# done
# done
# done

# wait

# for DATASET in meebits
# do
# for MODEL in DMF
# do
# for CONFIG in general
# do
# for SEED in 2022 2023 2024
# do

#     python main.py \
#         --model $MODEL \
#         --dataset $DATASET \
#         --config $CONFIG \
#         --seed $SEED &

# done
# done
# done
# done

# wait





# for DATASET in azuki
# do
# for MODEL in BPR
# do
# for CONFIG in general
# do
# for SEED in 2022 2023 2024
# do

#     python main.py \
#         --model $MODEL \
#         --dataset $DATASET \
#         --config $CONFIG \
#         --seed $SEED &

# done
# done
# done
# done

# wait

# for DATASET in bayc
# do
# for MODEL in BPR
# do
# for CONFIG in general
# do
# for SEED in 2024
# do

#     python main.py \
#         --model $MODEL \
#         --dataset $DATASET \
#         --config $CONFIG \
#         --seed $SEED &

# done
# done
# done
# done

# wait

# for DATASET in coolcats
# do
# for MODEL in BPR
# do
# for CONFIG in general
# do
# for SEED in 2022 2024
# do

#     python main.py \
#         --model $MODEL \
#         --dataset $DATASET \
#         --config $CONFIG \
#         --seed $SEED &

# done
# done
# done
# done

# wait

# for DATASET in doodles
# do
# for MODEL in BPR
# do
# for CONFIG in general
# do
# for SEED in 2022 2023 2024
# do

#     python main.py \
#         --model $MODEL \
#         --dataset $DATASET \
#         --config $CONFIG \
#         --seed $SEED &

# done
# done
# done
# done

# wait

# for DATASET in meebits
# do
# for MODEL in BPR
# do
# for CONFIG in general
# do
# for SEED in 2022 2023 2024
# do

#     python main.py \
#         --model $MODEL \
#         --dataset $DATASET \
#         --config $CONFIG \
#         --seed $SEED &

# done
# done
# done
# done

# wait








# for DATASET in azuki
# do
# for MODEL in ItemKNN
# do
# for CONFIG in general
# do
# for SEED in 2022 2023 2024
# do

#     python main.py \
#         --model $MODEL \
#         --dataset $DATASET \
#         --config $CONFIG \
#         --seed $SEED &

# done
# done
# done
# done

# wait

# for DATASET in bayc
# do
# for MODEL in ItemKNN
# do
# for CONFIG in general
# do
# for SEED in 2022 2023 2024
# do

#     python main.py \
#         --model $MODEL \
#         --dataset $DATASET \
#         --config $CONFIG \
#         --seed $SEED &

# done
# done
# done
# done

# wait

# for DATASET in coolcats
# do
# for MODEL in ItemKNN
# do
# for CONFIG in general
# do
# for SEED in 2022 2023 2024
# do

#     python main.py \
#         --model $MODEL \
#         --dataset $DATASET \
#         --config $CONFIG \
#         --seed $SEED &

# done
# done
# done
# done

# wait

# for DATASET in doodles
# do
# for MODEL in ItemKNN
# do
# for CONFIG in general
# do
# for SEED in 2022 2023 2024
# do

#     python main.py \
#         --model $MODEL \
#         --dataset $DATASET \
#         --config $CONFIG \
#         --seed $SEED &

# done
# done
# done
# done

# wait

# for DATASET in meebits
# do
# for MODEL in ItemKNN
# do
# for CONFIG in general
# do
# for SEED in 2022 2023 2024
# do

#     python main.py \
#         --model $MODEL \
#         --dataset $DATASET \
#         --config $CONFIG \
#         --seed $SEED &

# done
# done
# done
# done

# wait




# for DATASET in azuki
# do
# for MODEL in Pop
# do
# for CONFIG in general
# do
# for SEED in 2022 2023 2024
# do

#     python main.py \
#         --model $MODEL \
#         --dataset $DATASET \
#         --config $CONFIG \
#         --seed $SEED &

# done
# done
# done
# done

# wait

# for DATASET in bayc
# do
# for MODEL in Pop
# do
# for CONFIG in general
# do
# for SEED in 2022 2023 2024
# do

#     python main.py \
#         --model $MODEL \
#         --dataset $DATASET \
#         --config $CONFIG \
#         --seed $SEED &

# done
# done
# done
# done

# wait

# for DATASET in coolcats
# do
# for MODEL in Pop
# do
# for CONFIG in general
# do
# for SEED in 2022 2023 2024
# do

#     python main.py \
#         --model $MODEL \
#         --dataset $DATASET \
#         --config $CONFIG \
#         --seed $SEED &

# done
# done
# done
# done

# wait

# for DATASET in doodles
# do
# for MODEL in Pop
# do
# for CONFIG in general
# do
# for SEED in 2022 2023 2024
# do

#     python main.py \
#         --model $MODEL \
#         --dataset $DATASET \
#         --config $CONFIG \
#         --seed $SEED &

# done
# done
# done
# done

# wait

# for DATASET in meebits
# do
# for MODEL in Pop
# do
# for CONFIG in general
# do
# for SEED in 2022 2023 2024
# do

#     python main.py \
#         --model $MODEL \
#         --dataset $DATASET \
#         --config $CONFIG \
#         --seed $SEED &

# done
# done
# done
# done

# wait


