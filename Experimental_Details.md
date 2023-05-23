## Data sets 
We conducted experiments on four real-world NFT transaction datasets BAYC, Cool Cats, Doodles, and Meebits. To secure a sufficient number of test sets, 40\% of interactions from each user were randomly selected and used as a test set. As a result, the historical user-item interactions for each collection are split into 62\% training and 38\% testing on average, which are then equally and randomly divided into validation and test sets. 
We performed negative sampling to create user-specific pairwise preferences, designating user-purchased items as positive and selecting five negative samples for each positive sample based on popularity.<br>

<br>

## Baseline models (how experiments were done e.g. w features/wo features etc.)


## Hyperparameter details(our model, mgat, baseline models)
We optimised NFT-MARS using the Adam optimiser, a learning rate of 0.01, and 50 epochs.
Hyperparameter tuning involved a random search using Recall@50 as an indicator, with search ranges including the
dimensions (𝑑) of the graph’s final node representation [128, 512], loss alpha (𝛼) [0.1, 0.2], batch size [1024, 4096],
number of hops (𝐿) [1, 2, 3], and regularization weight [0.1, 0.001]. Baseline models’ hyperparameters were also tuned, in
regards to embedding size, learning rate, and dropout ratio. As each NFT collection has unique characteristics, separate
models were built and fitted for each collection, resulting in different best hyperparameters for each NFT collection. 
Best hyperparameter values for NFT-MARS are specified in below table. 
각 컬렉션 당 NFT-MARS의 hyperparameter details 써주기 영빈이가! 
MGAT was also optimised using Adam optimise. As for tuning hyperparameters foro MGAT, we fix the loss alpha (𝛼) to 0 test and compare the effectiveness of the multi-task learning and learning rate to 0.01, 

나머지는 영빈이의 몫...
