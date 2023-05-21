# 1. Data Description

## 1.1. Overview

The number of users, items and interactions for each collection used for the experiment.<br>
| collection | users | items | interactions |
|-------|------|------|-------------|
| BAYC  | 1230 | 6726 | 13737 |
| Coolcats | 1357 | 6824 | 14890 |
| Doodles | 804 | 4771 | 7250 |
| Meebits | 1184 | 6693 | 21104 |
<br>

## 1.2. Evaluating Data Density and Transaction Thresholds

We explore our dataset by evaluating its data density and transaction thresholds as determining the optimal transaction threshold is one of the critical steps in designing effective recommender systems. Below analyses were done with complete set of transactions.<br>

<br>

- **Number of transaction left after transaction threshold on users**
<div class="image-container">
    <img src="assets/user_trxns.png" alt="Plot" style="width:500px;height:300px;">
    
</div><br>
<br>

- **Number of transactions left after transaction threshold on items**
<div class="image-container">
    <img src="assets/item_trxns.png" alt="Plot" style="width:500px;height:300px;">
    
</div><br>
<br>

- **Number of users left after user threshold**
<div class="image-container">
    <img src="assets/user_num.png" alt="Plot" style="width:500px;height:300px;">
    
</div><br>
<br>

- **Number of items left after item threshold**
<div class="image-container">
    <img src="assets/item_num.png" alt="Plot" style="width:500px;height:300px;">
</div><br>
<br>   

## 1.3. Power law distribution<br>
By examining power law distributions, we gain insights into the relative popularity and concentration of tokens within the dataset. Therefore, we conduct an analysis of the power law distribution within our dataset and compare it with benchmark data. This comparison allows us to assess how our dataset aligns with established benchmarks and understand any deviations or unique characteristics present.<br>
<br>
1. Our dataset
- Bored Apes Yacht Club <br>
 <img src="assets/bayc.png" alt="Plot" style="width:500px;height:300px;"><br>
<br>

- Cool cats<br>
<img src="assets/coolcats.png" alt="Plot" style="width:500px;height:300px;">
<br>

- Doodles<br>
<img src="assets/doodles.png" alt="Plot" style="width:500px;height:300px;">
<br>

- Meebits<br>
<img src="assets/meebits.png" alt="Plot" style="width:500px;height:300px;">
<br>

2. Benchmark datasets<br>
<br>
- Amazon<br>
<img src="assets/amazon.png" alt="Plot" style="width:500px;height:300px;">
<br>

- MovieLens20M<br>
<img src="assets/movielens20m.png" alt="Plot" style="width:500px;height:300px;">
<br>

## 1.4. Token price movement<br>
In order to incorporate price movement labels (binary: 1 for upward movement, 0 for downward movement) for multi-task learning, we have conducted an examination of the price movements for each token. The columns in the table represent the following: the number of tokens for which the latest transaction price is greater than the initial transaction price, the number of tokens for which the latest transaction price is greater than the mean of the remaining transaction prices (excluding the last one), and the number of tokens for which the mean of the second half of the transaction prices is greater than the mean of the first half of the transaction prices. In the table, each row represents a specific token, and the values indicate the percentage of tokens that exhibit the respective price movements.<br>

<br>

| Token      | Latest Price > First Price | Latest Price >Avg. Price (Excluding Last) | Avg. Price (First Half) < Avg. Price (Last Half) |
|------------|-------------------------------|--------------|--------------|
| BAYC       | 59%                         | 61%       | 59% |
| Cool Cats  | 47%                         | 47%         | 50% |
| Doodles    | 58%                          | 48%        | 59% |
| Meebits    | 45%                          | 45%         |47% |


<br>

# 4. Item features preparation (이건 좀 겉도는 감이 있음 따로 빼던지 아님 놔두던지)
## Image
We employ Convolutional Auto-Encoder(CAE) to get representations from the NFT images. We standardise all images to a shape of `128 * 128 * 3`, where 3 represents the RGB colour spectrum. CAE model consists of an encoder and a decoder, both comprising of eight fully connected alyers, The encoder utilises a 33 convolutional kernel and 22 max pooling, while the decode employs a 33 convolutional kernel along with 22 upsampling.
All non-linear functions in the model are implemented using the ReLU activation function. The model is trained for 100 epochs, with the objective of minimising the Mean Squared Error (MSE) loss. The final image embeddings are obtained by employing only the encoder of the CAE, which reduces the data down to `8 * 8 * 1`
 size. After flattening the output, we receive a 64-dimension representation for each image. This refined data is them ready for subsequent modeling stages.<br>
<br>
## Text
The text data for each item is comprised of discrete words each describing each of the visual properties like ‘Background colour’, ‘body’, ‘outfit’, and ‘hair’. Items within the same collection share the same types of visual properties whereas they tend to vary across collections. Cool cats, for example, is a collection of blue cat avatars with each artwork possessing a unique body, hat, face, and outfit, whereas Bored Ape Yacht Club is a collection of monkey avatars having a slightly different types of properties like ‘fur’. Among all, we only have considered six types of properties with the fewest missing values for each collection apart from Cool cats, for which we considered all available 5 types of properties for generating item embeddings. We then processed each descriptive word into a 300-dimension word embedding. This was done by fetching the corresponding embeddings from a pre-trained Word2Vec model. If a particular word was not found in this model, we filled it with zero padding. It's worth noting that while a majority of visual attributes were described by a single word, those composed of multiplie words, like 'short red hair' for 'Hair', we used the sum of each word's embeddings instead. Each word embedding was concatenated with other embeddings related to the same item. As a result, each item's word embedding size ranged from 1500 to 1800, depending on the number of visual traits considered.<br>
<br>
## Price


## Transaction



