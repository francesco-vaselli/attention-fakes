from torch import nn
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from tqdm import tqdm

from dataset import FakesDataset
from model import FakesModel


if __name__=="__main__":
    batch_size = 64
    np_array = np.load("../extraction/test.npy")
    dataset = FakesDataset(np_array)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = FakesModel(3, 39, 64, 2, 39)
    # print total parameters
    params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {params}")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )
    FM = ConditionalFlowMatcher(sigma=0.001)
    loss_val = []
    for j in range(10):
        print(f"Epoch {j}")
        epoch_loss = 0
        with tqdm(total=len(dataloader), position=0,  dynamic_ncols=True, ascii=True) as pbar:
            for i, (features, pu, nfakes, _, key_padding_mask) in enumerate(dataloader) :
                optimizer.zero_grad()
                x0 = torch.randn(features.shape[0], 10, 39)
                t, xt, ut = FM.sample_location_and_conditional_flow(x0, features)
                vt = model(xt, pu, nfakes, t, key_padding_mask)

                # print(f"vtb: {vt.shape}")
                # print(f"utb: {ut.shape}")
                # loss is (vt - ut)**2 but I want to evaluate it only on the non-padded values
                # take the non-padded values from vt and ut
                good_idx = ~key_padding_mask
                vt = vt[:, 1:, :][good_idx]
                ut = ut[good_idx]
                loss = ((vt - ut)**2).mean()
                # print(f"vt: {vt.shape}")
                # print(f"ut: {ut.shape}")            
                # loss = ((vt[:, 1:, :] - ut)**2)[~key_padding_mask]
                loss = loss.mean()
                # print(f"loss: {loss}")
                loss.backward()
                epoch_loss += loss.item()
                optimizer.step()

                # Update the progress bar
                pbar.update(1)
                pbar.set_postfix({"Batch Loss": loss.item()})

        epoch_loss /= len(dataloader)
        print(f"Epoch loss: {epoch_loss}")
        loss_val.append(epoch_loss)
