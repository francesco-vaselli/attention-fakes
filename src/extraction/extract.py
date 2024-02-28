
#%%
import os
import ROOT
import yaml
import awkward as ak
import numpy as np


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
        extracted = extracted.Define(f"FJet_{col}", f"Jet_{col}[JetMask]")

    # check that FJet_col is present
    for col in columns:
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
    columns = [f"FJet_{col}" for col in columns]
    ak_array = ak.from_rdataframe(df, columns=columns)

    print(ak_array[1])
    # print shapes

    # pad the array with zeros
    max_len = 10
    ak_array = ak.fill_none(ak.pad_none(ak_array, max_len, clip=True), -999)

    # convert to numpy
    np_array = np.asarray(ak.concatenate(ak.unzip(ak_array[:, np.newaxis]), axis=1))
    return np_array

#%%
# def extract_jets(inputname, outputname, dict):
ROOT.EnableImplicitMT()
inputname= "/home/users/fvaselli/031C4ACC-DAA5-1640-9331-4B55F5716F04.root"
print(f"Processing {inputname}...")

# read columns from the yaml config
with open("columns.yaml", "r") as file:
    config = yaml.safe_load(file)

columns = config["reco_columns"]

d = ROOT.RDataFrame("Events", inputname)

d = extract_jet_features(d, columns)

np_array = get_numpy_array(d, columns)
print(np_array)

# swap the 1, 2 axes
np_array = np.swapaxes(np_array, 1, 2)
# outputname = "test.root"
# cols = [f"FJet_{col}" for col in columns]
# d.Snapshot("MJets", outputname, cols)
# cols = [f"FJet_{col}" for col in columns]
# np_cols = d.AsNumpy(columns=cols)

# print(np_cols)
# # now np_cols is a dictonary on Np_array. just get the values and you are done
# np_values = np_cols.values()
# print(np_values)

    # np_data = get_numpy_array(d, columns)



    # n_match, n_reco = dict["RECOJET_GENJET"]

    # n_match += d.Histo1D("MJet_ptRatio").GetEntries()
    # n_reco += d.Histo1D("Jet_pt").GetEntries()

    # dict["RECOJET_GENJET"] = (n_match, n_reco)

    # cols = jet_cond + reco_columns

    # d.Snapshot("MJets", outputname, cols)

    # print(f"{outputname} written")

# if __name__ == "__main__":
#     extract_jets("/home/users/fvaselli/031C4ACC-DAA5-1640-9331-4B55F5716F04.root", "test", {})
# %%
