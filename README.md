# RecSys2023

This is the origin Pytorch implementation of the paper **NFTs to MARS: Multi-modal multi-Attention Recommender System for NFTs**	





![model](assets/figure_model_architecture.png)





## Get started

1. Install Python 3.10.9

2. Download data. You can obtain all pre-processed data from [Google Drive](https://drive.google.com/drive/folders/12WeTJ6HzjGI0giirlu__PFSGtxno7cWU?usp=share_link).

3. Create a folder `dataset/collections` and place the downladed data in that location. 

4. Download requirement packages ***pip install -r requirements.txt***

5. Train the model. We provide the experiment scripts of all datasets in the file `run.sh`. You can reproduce the experiment results by: 

   ~~~bash
   ```
   bash run.sh
   ```
   ~~~

## Contribution

- We develop a model to address three unique challenges that NFT recommender systems face. Our method consists of three key components:
  1. **Graph attention** to handle extremely sparse user-item interactions
  2. **Multi-modal attention** to incorporate user-specific feature preferences
  3. **Multi-task learning** to address the dual nature of NFTs as artworks and investment assets
- We demonstrate the effectiveness of our model compared to various baseline models using the actual transaction data of NFTs collected directly from blockchain for four of the most popular NFT collections.
- We constructed a dataset by combining this transaction data with hand-crafted features, which can be used as a benchmark dataset for any NFT recommendation model.

## Baselines

In addition to "Our Model", there are two more folders: "Baseline models (MGAT)" and "Baseline models (Others).



## Acknowledgement

We appreciate the following github repos a lot for their valuable code base or datasets:

1. [RecBole](https://github.com/RUCAIBox/RecBole)
2. [MGAT](https://github.com/zltao/MGAT)
