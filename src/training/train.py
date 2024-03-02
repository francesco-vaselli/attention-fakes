from torch import nn
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from tqdm import tqdm
import yaml 
import sys
import os
import json

from dataset import FakesDataset
from model import FakesModel
from cfm import ModelWrapper, MyAlphaTTargetConditionalFlowMatcher
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import postprocess_jet_features, plot_jet_features


if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    batch_size = 64
    test_batch_size = 1000
    np_array = np.load("../extraction/test.npy")
    dataset = FakesDataset(np_array)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # split in train and test
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    model = FakesModel(input_size=3, feature_size=39, embed_size=64, n_heads=2)
    model = model.to(device)

    # print total parameters
    params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {params}")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )
    
    FM = MyAlphaTTargetConditionalFlowMatcher(sigma=0.0001, alpha=1)
    loss_val = []
    for j in range(0):
        print(f"Epoch {j}")
        epoch_loss = 0
        with tqdm(total=len(dataloader), position=0,  dynamic_ncols=True, ascii=True) as pbar:
            for i, (features, pu, nfakes, _, key_padding_mask) in enumerate(dataloader) :
                features = features.to(device)
                pu = pu.to(device)
                nfakes = nfakes.to(device)
                key_padding_mask = key_padding_mask.to(device)

                optimizer.zero_grad()
                x0 = torch.randn(features.shape[0], features.shape[1], 39).to(device)
                t, xt, ut = FM.sample_location_and_conditional_flow(x0, features)
                vt = model(xt, pu, nfakes, t, key_padding_mask)

                # print(f"vtb: {vt.shape}")
                # print(f"utb: {ut.shape}")

                good_idx = ~key_padding_mask
                vt = vt[good_idx]
                ut = ut[good_idx]
                loss = ((vt - ut)**2).mean()
                # print(f"vt: {vt.shape}")
                # print(f"ut: {ut.shape}")            
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

    sampler = ModelWrapper(model, device=device)
    t_span = torch.linspace(0, 1, 100).to(device)
    # sample points in test
    with torch.no_grad():
        samples = []
        nfakes_list = []
        masks = []
        features_list = []
        with tqdm(total=len(test_dataloader), position=0,  dynamic_ncols=True, ascii=True) as pbar:
            for i, (features, pu, nfakes, _, _) in enumerate(test_dataloader):
                # we construct the mask at runtime to simulate inference in a real case
                key_padding_mask = torch.arange(sampler.max_len).expand(len(pu), sampler.max_len)
                # for each entry, put the mask to 1 where idx is greater than nfakes
                key_padding_mask = torch.where(key_padding_mask < nfakes.unsqueeze(1), torch.zeros_like(key_padding_mask), torch.ones_like(key_padding_mask))
                sample = sampler.generate(pu, nfakes, key_padding_mask, t_span=t_span).cpu().numpy()
                samples.append(sample)
                nfakes_list.append(nfakes.cpu().numpy())
                masks.append(key_padding_mask.cpu().numpy())
                features_list.append(features.cpu().numpy())
                pbar.update(1)
        
        samples = np.concatenate(samples, axis=0)
        nfakes = np.concatenate(nfakes_list, axis=0)
        masks = np.concatenate(masks, axis=0)
        features = np.concatenate(features_list, axis=0)
        print(f"samples: {samples.shape}")

    with open("../extraction/columns.yaml", "r") as file:
        config = yaml.safe_load(file)
        preprocess_ops = config["preprocess_ops"]

    # read the scaler_params.json
    with open("../extraction/scaler_params.json", "r") as file:
        scaler_params = json.load(file)    

    # postprocess the samples and the features
    samples = postprocess_jet_features(samples, preprocess_ops, scaler_params, masks)
    features = postprocess_jet_features(features, preprocess_ops, scaler_params, masks)

    # plot the features
    for i in range(10):
        plot_jet_features(features, samples, masks, i, config["reco_columns"], "test")
