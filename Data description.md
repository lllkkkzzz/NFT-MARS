# 1. Data Description

## 1.1. Overview

number of users, items and interactions for each collection
|       | user | item | interaction |
|-------|------|------|-------------|
| BAYC  | 1230 | 6726 | 13737 |
| Coolcats | 1357 | 6824 | 14890 |
| Doodles | 804 | 4771 | 7250 |
| Meebits | 1184 | 6693 | 21104 |



## 1.2. Data Sparsity
### 1.2.1. Bored Apes Yacht Club 

transaction sparsity after user/item cut

number of items left after cut

number of users left after cut

### 1.2.2. Cool cats

transaction sparsity after user/item cut

number of items after cut

number of users after cut


### 1.2.3. Doodles

transaction sparsity after user/item cut

number of items after cut

number of users after cut


### 1.2.4. Meebits

transaction sparsity after user/item cut

number of items after cut

number of users after cut


# 2. Graph connectivity
## 2.1. Bored Apes Yacht Club 


## 2.2. Cool cats


## 2.3. Doodles


## 2.4. Meebits




# 2. Item features preparation

## 2.1. Image
We employ Convolutional Auto-Encoder(CAE) to get representations from the NFT images. We standardise all images to a shape of 128*128*3, where 3 represents the RGB colour spectrum. CAE model consists of an encoder and a decoder, both comprising of eight fully connected alyers, The encoder utilises a 33 convolutional kernel and 22 max pooling, while the decode employs a 33 convolutional kernel along with 22 upsampling.
All non-linear functions in the model are implemented using the ReLU activation function. The model is trained for 100 epochs, with the objective of minimising the Mean Squared Error (MSE) loss. The final image embeddings are obtained by employing only the encoder of the CAE, which reduces the data down to 8*8*1 size. After flattening the output, we receive a 64-dimension representation for each image. This refined data is them ready for subsequent modeling stages.

## 2.2. Text
The text data for each item is comprised of discrete words each describing each of the visual properties like ‘Background colour’, ‘body’, ‘outfit’, and ‘hair’. Items within the same collection share the same types of visual properties whereas they tend to vary across collections. Cool cats, for example, is a collection of blue cat avatars with each artwork possessing a unique body, hat, face, and outfit, whereas Bored Ape Yacht Club is a collection of monkey avatars having a slightly different types of properties like ‘fur’. Among all, we only have considered six types of properties with the fewest missing values for each collection apart from Cool cats, for which we considered all available 5 types of properties for generating item embeddings. We then processed each descriptive word into a 300-dimension word embedding. This was done by fetching the corresponding embeddings from a pre-trained Word2Vec model. If a particular word was not found in this model, we filled it with zero padding. It's worth noting that while a majority of visual attributes were described by a single word, those composed of multiplie words, like 'short red hair' for 'Hair', we used the sum of each word's embeddings instead. Each word embedding was concatenated with other embeddings related to the same item. As a result, each item's word embedding size ranged from 1500 to 1800, depending on the number of visual traits considered.

## 2.3. Price

## 2.4. Transaction



