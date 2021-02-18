# Neural Machine Translation
English-to-German Neural Machine Translation (NMT) model using Long Short-Term Memory (LSTM) networks with Attention.

## Introduction

Machine translation is an important task in natural language processing and could be useful not only for translating one 
language to another but also for word sense disambiguation (e.g. determining whether the word "bank" refers to the financial bank, 
or the land alongside a river). Implementing this using just a Recurrent Neural Network (RNN) with LSTMs can work for short to medium length 
sentences but can result in vanishing gradients for very long sequences. 
To solve this, you will be adding an attention mechanism to allow
 the decoder to access all relevant parts of the input sentence regardless of its length.

## Libraries
I will use the [Trax](https://github.com/google/trax) library created and maintained by the [Google Brain team](https://research.google/teams/brain/) to do most of the heavy lifting. 
It provides submodules to fetch and process the datasets, as well as build and train the model.

## Dataset

I will just use a small dataset from [Opus](https://opus.nlpl.eu/), a growing collection of translated texts from the web. 
Particularly, I will get an English to German translation subset specified as opus/medical which has medical related texts. 
If storage is not an issue, you can opt to get a larger corpus such as the English to German translation dataset from [ParaCrawl](https://paracrawl.eu/), 
a large multi-lingual translation dataset created by the European Union. 
Both of these datasets are available via [Tensorflow Datasets (TFDS)](https://www.tensorflow.org/datasets) and we 
can browse through the other available datasets [here](https://www.tensorflow.org/datasets/catalog/overview). 

I have downloaded the data in the **data/** directory of my workspace. I haven't uploaded the data folder containing my dataset to my repository 
but you can easily access this dataset from TFDS with `trax.data.TFDS`. 
The result is a python generator function yielding tuples. Use the keys argument to select what appears at which position in the tuple. 
For example, keys=('en', 'de') below will return pairs as (English sentence, German sentence).