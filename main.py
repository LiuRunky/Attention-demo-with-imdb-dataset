import torch
import numpy as np
import torch.nn as nn

from dataset_utils import *
from attention import NaiveMultiHeadAttention


class TorchClassificationModel(nn.Module):
    def __init__(self, model_dim, heads, max_features, max_len):
        super(TorchClassificationModel, self).__init__()
        self.model_dim = model_dim
        self.heads = heads
        self.max_features = max_features
        self.max_len = max_len
        self.embed = nn.Embedding(num_embeddings=max_features, embedding_dim=model_dim)
        # self.attn = nn.MultiheadAttention(embed_dim=model_dim, num_heads=heads)
        self.attn = NaiveMultiHeadAttention(embed_dim=model_dim, num_heads=heads)
        self.linear = nn.Linear(in_features=model_dim, out_features=1)

    def forward(self, input):
        embeddings = self.embed(input)
        attn_output, _ = self.attn(query=embeddings, key=embeddings, value=embeddings)
        output = torch.mean(attn_output.view(attn_output.size(0), attn_output.size(2), -1), dim=2)
        output = nn.Dropout(0.5)(output)
        output = self.linear(output)
        output = nn.Sigmoid()(output).view(-1)
        return output


def sequence_padding(input, max_len, position="front", value=0):
    """
    Add padding to Numpy array of lists with shape [batch_size, sequence_length (not fixed)].

    Args:
        input: Numpy array of lists. A batch of sequences.
        max_len: integer. Maximal length of the padded sequence.
        position (Optional): str. Position of the padded value, either "front" or "rear".
        value (Optional): integer. Value to be padded into sequence.

    Returns:
        output: torch.Tensor. Padded sequence with shape [batch_size, max_len] and dtype=torch.int64.
    """
    assert position == "front" or position == "rear"

    batch_size = len(input)
    output = np.full((batch_size, max_len), value)
    for i in range(batch_size):
        sequence_len = min(len(input[i]), max_len)
        if position == "front":
            output[i, :sequence_len] = np.array(input[i][:sequence_len])
        else:
            output[i, -sequence_len:] = np.array(input[i][:sequence_len])
    return torch.Tensor(output).to(torch.int64)


if __name__ == "__main__":
    epochs = 10
    batch_size = 32
    model_dim = 128
    heads = 8
    max_features = 20000
    max_len = 80

    (x_train, y_train), (x_test, y_test) = load_imdb_dataset(num_words=max_features, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device)
    model = TorchClassificationModel(model_dim=model_dim, heads=heads, max_features=max_features, max_len=max_len)
    model = model.to(device)

    optim = torch.optim.Adam(model.parameters())
    loss_fn = nn.BCELoss()

    for epoch in range(epochs):
        train_correct = 0
        for i in range(0, len(x_train) - batch_size, batch_size):
            optim.zero_grad()

            pad_input = sequence_padding(input=x_train[i: i+batch_size], max_len=max_len, position="rear", value=0)
            pad_input = pad_input.to(device)
            target = torch.Tensor(y_train[i: i+batch_size]).to(dtype=torch.float, device=device)

            model_output = model(pad_input)
            train_correct += ((model_output > 0.5) == target).sum()

            loss = loss_fn(model_output, target)
            loss.backward()
            optim.step()

        test_correct = 0
        with torch.no_grad():
            for i in range(0, len(x_test) - batch_size, batch_size):
                pad_input = sequence_padding(input=x_test[i: i+batch_size], max_len=max_len, position="rear", value=0)
                pad_input = pad_input.to(device)
                target = torch.Tensor(y_test[i: i+batch_size]).to(dtype=torch.float, device=device)

                model_output = model(pad_input)
                test_correct += ((model_output > 0.5) == target).sum()

        print("epoch = {}".format(epoch))
        print("train_acc = {}   test_acc = {}".format(train_correct / len(x_train), test_correct / len(x_test)))
