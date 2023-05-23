## Data sets 
We conducted experiments on four real-world NFT transaction datasets BAYC, Cool Cats, Doodles, and Meebits. To secure a sufficient number of test sets, 40\% of interactions from each user were randomly selected and used as a test set. As a result, the historical user-item interactions for each collection are split into 62\% training and 38\% testing on average, which are then equally and randomly divided into validation and test sets. 
We performed negative sampling to create user-specific pairwise preferences, designating user-purchased items as positive and selecting five negative samples for each positive sample based on popularity.<br>

<br>

## Baseline models (how experiments were done e.g. w features/wo features etc.)
OIn our experiments, we compared our proposed NFT-MARS model, which uniquely combines a graph-based collaborative filtering (CF) approach with a content-based method incorporating both user and item features, against a range of baseline models. These baseline models consisted of Pop, ItemKNN, BPR, DMF, NeuMF, and LightGCN, which were executed without any side information like user or item features. On the other hand, we also experimented with models like FM, DeepFM, WideDeep, DCN, AutoInt, and MGAT, utilizing side information. The experimentation was carried out using RecBole. The goal of these comparisons is to compare the effectiveness of different methods in the context of our task.


## Hyperparameter details(our model, mgat, baseline models)
We optimised NFT-MARS using the Adam optimiser, a learning rate of 0.01, and 50 epochs.
Hyperparameter tuning involved a random search using Recall@50 as an indicator, with search ranges including the
dimensions (𝑑) of the graph’s final node representation [128, 512], loss alpha (𝛼) [0.1, 0.2], batch size [1024, 4096],
number of hops (𝐿) [1, 2, 3], and regularisation weight [0.1, 0.001]. Best hyperparameter values for NFT-MARS are specified in below table.<br>
<br>

| collection | seed | dimension (𝑑) | loss alpha (𝛼) | batch size | number of hops (𝐿) | regularisation weight
|-------|------|------|-------------|-------------|-------------|-------------|
| BAYC  | 2023 | 128 | 0.2 | 1024 | 2 | 0.1 |
| Coolcats | 2024 | 512 | 0.2 | 1024 | 1 | 0.1 |
| Doodles | 2022 | 512 | 0.1 | 1024 | 3 | 0.001 |
| Meebits | 2022 | 512 | 0.1 | 1024 | 1 | 0.001 |

<br>
Same optimiser, Adam, was also used for optimisation of MGAT model. As for tuning hyperparameters for MGAT, we fix the loss alpha (𝛼) to 0 test and compare the effectiveness of the multi-task learning and learning rate to 0.01. We then select the dimensions (𝑑) of the graph’s final node representation from [128, 512], batch size from [1024, 4096], regularisation weight from [0.1, 0.001], and number of hops (𝐿) from [1, 2, 3]. Best hyperparameter values for MGAT are specified in below table.<br>
<br>

| collection | seed | dimension (𝑑) | batch size | number of hops (𝐿) | regularisation weight
|-------|------|------|-------------|-------------|-------------|-------------|
| BAYC  | 2022 | 128 | 4096 | 1 | 0.1 |
| Coolcats | 2024 | 512 | 1024 | 1 | 0.001 |
| Doodles | 2023 | 512 | 1024 | 1 | 0.001 |
| Meebits | 2024 | 128 | 1024 | 1 | 0.001 |

<br>
Baseline models’ hyperparameters were also tuned, in regards to embedding size, learning rate, and dropout ratio. Specific details like the optimal hyperparameter values, hyperparameter search range are specified in [hyper](Baseline models (Others)/hyper) folder.

