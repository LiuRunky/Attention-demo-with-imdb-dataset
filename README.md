## Attention demo with imdb dataset

PyTorch implementation of naive multi-head self-attention, with correctness roughly validated on imdb dataset.

The details of environmental settings are given below. However, we assume almost any version is acceptable.

```
python=3.6
numpy=1.19.5
pytorch=1.10.2
```

### Introduction

The repository has the following structure:

```
Attention-demo-with-imdb-dataset
├───attention.py
├───dataset_utils.py
├───main.py
└───dataset
    ├───"PUT DATASET HERE
    └───(imdb.npz)
```

In `attntion.py`, a naive implementation of multi-head attention is presented.
The class `NaiveMultiHeadAttention` receives two arguments: `embed_dim` and `num_heads`,
which are in line with `nn.MultiheadAttention`. This class does not implement attention mask.

In `dataset_utils`, an adapted version of `keras.datasets.imdb.load_data()` is implemented,
so that one does not need the complicated pre-processing procedure to utilize the imdb dataset.

In `main.py`, a simple neural network is implemented to perform the sentiment classification task.
The network sends a sequence of word embeddings into multi-head attention to calculate self-attention,
then averages the sequence to only one embedding.
Later on, this embedding passes through `nn.Dropout` and `nn.Linear` to produce a scalar output.
We finally compute `nn.Sigmoid` on this scalar output to make the prediction.

**Disclaimer:** This may not be the ideal structure of the classification model,
but we are here to briefly verify the correctness of class `NaiveMultiHeadAttention`.

### Preparation

You can either download the source code or use `git clone` in terminal:

```
git clone https://github.com/LiuRunky/Attention-demo-with-imdb-dataset.git
```

After that, please download imdb dataset from https://storage.googleapis.com/tensorflow/tf-keras-datasets/imdb.npz
and move `imdb.npz` into the `dataset` folder.

### Execution

Simply execute the following command to train the classification model.

```
python main.py
```

You are free to compare the results with official implementation of `nn.MultiheadAttention`.
In `main.py`, uncommenting `Line 17` and commenting `Line 18` to perform such a switch.

### Sample Output

When using `NaiveMultiHeadAttention`, the sample output is shown below.

```
device = cuda
epoch = 0
train_acc = 0.7029599547386169   test_acc = 0.7763199806213379
epoch = 1
train_acc = 0.8291999697685242   test_acc = 0.7925199866294861
epoch = 2
train_acc = 0.8709200024604797   test_acc = 0.7871599793434143
epoch = 3
train_acc = 0.9053599834442139   test_acc = 0.783079981803894
epoch = 4
train_acc = 0.9315199851989746   test_acc = 0.7753999829292297
epoch = 5
train_acc = 0.9527999758720398   test_acc = 0.7646399736404419
epoch = 6
train_acc = 0.9659599661827087   test_acc = 0.7613599896430969
epoch = 7
train_acc = 0.9765999913215637   test_acc = 0.7606399655342102
epoch = 8
train_acc = 0.982479989528656   test_acc = 0.7608799934387207
epoch = 9
train_acc = 0.9846400022506714   test_acc = 0.7582399845123291
```

As a comparison, we also display the sample output when using `nn.MultiheadAttention`.

```
device = cuda
epoch = 0
train_acc = 0.5635600090026855   test_acc = 0.6651999950408936
epoch = 1
train_acc = 0.7331599593162537   test_acc = 0.7297599911689758
epoch = 2
train_acc = 0.7924799919128418   test_acc = 0.7414799928665161
epoch = 3
train_acc = 0.8296399712562561   test_acc = 0.747439980506897
epoch = 4
train_acc = 0.8547599911689758   test_acc = 0.7495999932289124
epoch = 5
train_acc = 0.8782399892807007   test_acc = 0.7443999648094177
epoch = 6
train_acc = 0.8953199982643127   test_acc = 0.7450000047683716
epoch = 7
train_acc = 0.9125199913978577   test_acc = 0.7321599721908569
epoch = 8
train_acc = 0.9247999787330627   test_acc = 0.7230799794197083
epoch = 9
train_acc = 0.9345999956130981   test_acc = 0.7319599986076355
```
