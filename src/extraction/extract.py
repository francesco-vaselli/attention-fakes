
#%%
import os
import ROOT
import yaml
import awkward as ak
import numpy as np
import json
from sklearn.preprocessing import StandardScaler


ROOT.gInterpreter.Declare('''
#include <ROOT/RVec.hxx>

// Overload for float RVec
auto pad_vector(ROOT::VecOps::RVec<float> & Rvec, int max_len) {
    float pad_value = -999.0f; // Hardcoded padding value for float
    ROOT::VecOps::RVec<float> padded(max_len, pad_value);
    for (size_t i = 0; i < std::min(Rvec.size(), static_cast<size_t>(max_len)); i++) {
        padded[i] = Rvec[i];
    }
    return padded;
}

// Overload for int RVec
auto pad_vector(ROOT::VecOps::RVec<int> & Rvec, int max_len) {
    int pad_value = -999; // Hardcoded padding value for int
    ROOT::VecOps::RVec<int> padded(max_len, pad_value);
    for (size_t i = 0; i < std::min(Rvec.size(), static_cast<size_t>(max_len)); i++) {
        padded[i] = Rvec[i];
    }
    return padded;
}

// Overload for double RVec  
auto pad_vector(ROOT::VecOps::RVec<double> & Rvec, int max_len) {
    double pad_value = -999.0; // Hardcoded padding value for double
    ROOT::VecOps::RVec<double> padded(max_len, pad_value);
    for (size_t i = 0; i < std::min(Rvec.size(), static_cast<size_t>(max_len)); i++) {
        padded[i] = Rvec[i];
    }
    return padded;
} 

// Overload for unsigned char RVec
auto pad_vector(ROOT::VecOps::RVec<unsigned char> & Rvec, int max_len) {
    unsigned char pad_value = 255; // Hardcoded padding value for unsigned char
    ROOT::VecOps::RVec<unsigned char> padded(max_len, pad_value);
    for (size_t i = 0; i < std::min(Rvec.size(), static_cast<size_t>(max_len)); i++) {
        padded[i] = Rvec[i];
    }
    return padded;
}
''')

def extract_jet_features(df, columns):
    """for going from GenJet to reco jet

    Args:
        df (rdataframe): original rdataframe (should be cleaned? to be decided)

    Returns:
        rdataframe: rdataframe with new features
    """

    # for each column in column, do
    # df.Define("FJet_column", "Jet_column[condition]") plus additional conditions
    extracted = df.Define("JetMask", "Jet_genJetIdx < 0 || Jet_genJetIdx > nGenJet")

    columns_dtypes = {}
    for col in columns:
        if col != "Pileup_nTrueInt":
            extracted = extracted.Define(f"FJet_{col}", f"Jet_{col}[JetMask]")

    # check that FJet_col is present
    for col in columns:
        if col != "Pileup_nTrueInt":
            assert f"FJet_{col}" in extracted.GetColumnNames(), f"FJet_{col} not present in the columns!!"

    # for each event we allow for max 10 fake jets, we pad those with less
    # max_len = 10
    # for col in columns:
    #     extracted = extracted.Redefine(f"FJet_{col}", f"pad_vector(FJet_{col}, {max_len})")

    return extracted

def get_numpy_array(df, columns):
    """gets the columns as numpy array from the rdataframe
        for each event we allow for max 10 fake jets, we pad those with less 

    Args:
        df (RDataFrame): RDataFrame
        column (str): column name

    Returns:
        np.array: numpy array
    """
    columns = ["Pileup_nTrueInt"]+[f"FJet_{col}" for col in columns if col != "Pileup_nTrueInt"]
    ak_array = ak.from_rdataframe(df, columns=columns)

    pu = np.asarray(ak_array["Pileup_nTrueInt"])
    ak_array = ak.without_field(ak_array, "Pileup_nTrueInt")

    # pad the array with zeros
    max_len = 10
    ak_array = ak.fill_none(ak.pad_none(ak_array, max_len, clip=True), -999)

    # convert to numpy
    np_array = np.asarray(ak.concatenate(ak.unzip(ak_array[:, np.newaxis]), axis=1))

    # now reshape to have shape (len, 1)
    pu = pu[:, np.newaxis]
    print(pu.shape)
    # pad the pileup to have shape (len, 10)
    pu = np.repeat(pu, max_len, axis=1)
    print(pu.shape)
    # reshape into (len, 1, 10)
    pu = pu[:, np.newaxis, :]

    # now concat all to get (len, 1+39, 10)
    np_array = np.concatenate((pu, np_array), axis=1)
    return np_array

def preprocess_jet_features(np_array, preprocess_ops):
    """preprocess the jet features
        shape is (len, 10, 40) but most of the 10 jets are padded so we do not preprocess them

    """
    scaler = StandardScaler()
    scaler_params = []
    for (feature, col) in zip(range(np_array.shape[2]), preprocess_ops.keys()):
        flat_data = np_array[:, :, feature].flatten()
    
        # Mask to identify non-padded values
        mask = flat_data != -999
        for op in preprocess_ops[col]:
            if op == None:
                continue
            elif op == "smear":
                # add uniform noise in [-0.5, 0.5] to the non-padded values
                flat_data[mask] += np.random.rand(np.sum(mask)) - 0.5
            elif op == "scale":
                scaled_values = scaler.fit_transform(flat_data[mask].reshape(-1, 1)).flatten()
                flat_data[mask] = scaled_values
                mean = scaler.mean_[0]
                scale = scaler.scale_[0]
                scaler_params.append({'feature': col, 'mean': mean, 'std': scale})

        np_array[:, :, feature] = flat_data.reshape(np_array.shape[0], np_array.shape[1])

    # Save to a JSON file
    with open('scaler_params.json', 'w') as f:
        json.dump(scaler_params, f, indent=4)

    return np_array

def repeat_and_copy(np_array):
    """repeats each array by the number of non-padded jets,
    """
    data = np_array
    pu = data[:, :, 0]
    features = data[:, :, 1:]
    nfakes = (features[:, :, 35] != -999).sum(axis=1)

    # Repeat each sequence and its features
    repeated_features = np.repeat(features, repeats=10, axis=0)
    repeated_pu = np.repeat(pu, repeats=10, axis=0)

    input_sequences = []
    target = []
    pu_vals = []
    for j in range(10):
        target.append(repeated_features[j::10, j, :])
        # print(repeated_features[j::10, j, :].shape)
        pad_features = repeated_features[j::10, :, :]
        pad_features[:, j, :] = -999
        input_sequences.append(pad_features)
    
    # Convert to numpy arrays for efficiency in indexing during __getitem__
    input_sequences = np.array(input_sequences)
    target = np.array(target)

    # concat repeated_pu and input_sequences
    input_sequences = np.concatenate((repeated_pu, input_sequences), axis=2)

    return input_sequences, target

    
#%%
def extract_jets(inputname, outputname):
    ROOT.EnableImplicitMT()
    inputname= "/home/users/fvaselli/031C4ACC-DAA5-1640-9331-4B55F5716F04.root"
    print(f"Processing {inputname}...")

    # read columns from the yaml config
    with open("columns.yaml", "r") as file:
        config = yaml.safe_load(file)

    columns = config["reco_columns"]

    d = ROOT.RDataFrame("Events", inputname)

    d = extract_jet_features(d, columns)

    # columns = ["Pileup_nTrueInt"]+[f"FJet_{col}" for col in columns if col != "Pileup_nTrueInt"]
    # ak_array = ak.from_rdataframe(d, columns=columns)

    np_array = get_numpy_array(d, columns)
    print(np_array)

    # # swap the 1, 2 axes
    np_array = np.swapaxes(np_array, 1, 2)
    print(np_array.shape)

    # check (len, :, 35), if all are -999, then remove the row
    p_T = np_array[:, :, 35]
    mask = p_T != -999
    mask = np.any(mask, axis=1)
    np_array = np_array[mask]
    print(np_array.shape)
    print(np_array[10])

    # preprocess the data
    preprocess_ops = config["preprocess_ops"]
    np_array = preprocess_jet_features(np_array, preprocess_ops)
    print(np_array[10])

    # repeat and copy
    input_sequences, target = repeat_and_copy(np_array)
    print(input_sequences[2, 0, :])
    print(target[2, 0, :])

    # save the array
    # np.save(f"{outputname}.npy", np_array)
# outputname = "test.root"
# cols = [f"FJet_{col}" for col in columns]
# d.Snapshot("MJets", outputname, cols)
# cols = [f"FJet_{col}" for col in columns]
# np_cols = d.AsNumpy(columns=cols)

if __name__ == "__main__":
    extract_jets("/home/users/fvaselli/031C4ACC-DAA5-1640-9331-4B55F5716F04.root", "test")
# %%
