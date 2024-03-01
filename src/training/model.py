from torch import nn
import torch
from dataset import FakesDataset
from torch.utils.data import Dataset, DataLoader
import numpy as np

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class FakesModel(nn.Module):
    def __init__(self, input_size, feature_size, embed_size, n_heads, output_size):
        super(FakesModel, self).__init__()
        self.embed_size = embed_size
        self.input1 = nn.Linear(input_size, embed_size)
        self.input2 = nn.Linear(feature_size, embed_size)

        self.mha1 = nn.MultiheadAttention(embed_size, n_heads, batch_first=True)
        self.mha2 = nn.MultiheadAttention(embed_size, n_heads, batch_first=True)
        # attention mask of shape (11, 11)
        attention_mask = torch.triu(torch.ones(12, 12)).clone().detach().bool()
        self.attention_mask = attention_mask[1:, :-1]
        # print(f"attention_mask: {self.attention_mask.shape}")
        # print(f"attention_mask: {self.attention_mask}")
        self.ln1 = nn.LayerNorm(embed_size)
        # a resnet block
        self.ffw1 = FeedFoward(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)
        self.ffw2 = FeedFoward(embed_size)
        self.ln3 = nn.LayerNorm(embed_size)
        self.ffw3 = FeedFoward(embed_size)
        self.ln4 = nn.LayerNorm(embed_size)
        self.ffw4 = FeedFoward(embed_size)
        self.ln_f = nn.LayerNorm(embed_size)
        self.output = nn.Linear(embed_size, output_size)

    def forward(self, x, pu, nfakes, toss, key_padding_mask):
        # cat pu, nfakse and toss 
        start = torch.cat([pu.view(-1, 1), nfakes.view(-1, 1), toss.view(-1, 1)], dim=1)
        # input1
        start = self.input1(start).view(-1, 1, self.embed_size)
        # print(f"start: {start.shape}")
        # input2
        # print(f"xb: {x.shape}")
        x = self.input2(x)
        # print(f"x: {x.shape}")
        # cat start and x
        x = torch.cat([start, x], dim=1)
        # print(f"xn: {x.shape}")
        # expand key padding mask with one False at the beginning of each sequence
        key_padding_mask = torch.cat([torch.zeros(key_padding_mask.shape[0], 1, dtype=torch.bool), key_padding_mask], dim=1)
        # print(f"key_padding_mask: {key_padding_mask.shape}")
        # ln1
        x = self.ln1(x)
        # mha
        x = x + self.mha1(x, x, x, key_padding_mask=key_padding_mask, attn_mask=self.attention_mask)[0]
        # print(f"mha1: {x.shape}")
        # print(f"mha1: {x}")
        # ffw1
        x = x + self.ffw1(self.ln2(x))
        x = self.ln3(x)
        # mha2
        x = x + self.mha2(x, x, x)[0]
        # ffw2
        x = x + self.ffw2(self.ln4(x))
        x = self.ln_f(x)
        # output
        x = self.output(x)
        return x

if __name__ == "__main__":

    np_array = np.load("../extraction/test.npy")
    dataset = FakesDataset(np_array)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    model = FakesModel(3, 39, 64, 2, 39)
    # print total parameters
    params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {params}")
    outs = []
    for i, (features, pu, nfakes, toss, key_padding_mask) in enumerate(dataloader):
        out = model(features, pu, nfakes, toss, key_padding_mask)
        # print(f"batch {i}")
        # print(f"out: {out.shape}")
        # print(f"out: {out}")
        outs.append(out)
        if i == 2:
            break
    
    outs = torch.cat(outs, dim=0)
    print(f"outs: {outs.shape}")