from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch


class FakesDataset(Dataset):
    def __init__(self, np_array):
        self.data = np_array
        self.pu = self.data[:, 0, 0] # review this maybe its [:, :, 0]
        self.features = self.data[:, :, 1:] 

        # get the Number of fakes in the event
        self.nfakes = (self.features[:, :, 35] != -999).sum(axis=1)
        # toss a random t in 0-1 for each event
        self.toss = np.random.rand(len(self.data))

        # # Repeat each sequence and its features to create shifted versions
        # self.repeated_features = np.repeat(self.features, repeats=self.features.shape[1], axis=0)
        # self.repeated_pu = np.repeat(self.pu, repeats=self.features.shape[1], axis=0)
        # self.repeated_nfakes = np.repeat(self.nfakes, repeats=self.features.shape[1], axis=0)
        
        # # Calculate shifts for input sequences
        # input_sequences = []
        # target_sequences = []
        # pu_vals = []
        # nfakes_vals = []
        # for pos in range(self.features.shape[1]):
        #     # Shift features to create input sequences with increasing padding
        #     shifted_features = np.roll(self.repeated_features, shift=shift, axis=1)
        #     # After shifting, replace the 'shifted in' values with padding
        #     shifted_features[:, :shift, :] = -999
            
        #     # Create target sequences
        #     targets = np.roll(self.repeated_features, shift=-1, axis=1)[:, 0, :]
            
        #     # Filter out sequences that would only contain padding after the shift
        #     valid_idx = np.any(shifted_features != -999, axis=(1, 2))
            
        #     input_sequences.append(shifted_features[valid_idx])
        #     target_sequences.append(targets[valid_idx])
        #     pu_vals.append(self.repeated_pu[valid_idx])
        #     nfakes_vals.append(self.repeated_nfakes[valid_idx])
        
        # # Convert to numpy arrays for efficiency in indexing during __getitem__
        # self.input_sequences = np.array(input_sequences)
        # self.target_sequences = np.array(target_sequences)
        # self.pu = np.array(pu_vals)
        # self.nfakes = np.array(nfakes_vals)
        
        # define the key_padding_mask for MHA, a mask of shape (N,S)
        self.key_padding_mask = self.features[:, :, 35] == -999

        # cast to torch tensors
        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.pu = torch.tensor(self.pu, dtype=torch.float32)
        self.nfakes = torch.tensor(self.nfakes, dtype=torch.float32)
        self.toss = torch.tensor(self.toss, dtype=torch.float32)
        self.key_padding_mask = torch.tensor(self.key_padding_mask, dtype=torch.bool)


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.features[idx], self.pu[idx], self.nfakes[idx], self.toss[idx], self.key_padding_mask[idx]
    

if __name__ == "__main__":

    np_array = np.load("../extraction/test.npy")
    dataset = FakesDataset(np_array)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    for i, (features, pu, nfakes, toss, key_padding_mask) in enumerate(dataloader):
        print(f"batch {i}")
        print(f"features: {features.shape}")
        print(f"features: {features}")
        print(f"pu: {pu.shape}")
        print(f"pu: {pu}")
        print(f"nfakes: {nfakes.shape}")
        print(f"nfakes: {nfakes}")
        print(f"toss: {toss.shape}")
        print(f"toss: {toss}")
        print(f"key_padding_mask: {key_padding_mask.shape}")
        print(f"key_padding_mask: {key_padding_mask}")
        print()
        if i == 2:
            break