import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import wasserstein_distance
import os


def postprocess_jet_features(data, preprocess_ops, scaler_params, pad_mask):
    """
    Postprocess the data to return the original data
    """
    mask = ~pad_mask
    # mask is of shape (len, 10)
    # need to get it to (len, 10, data.shape[2])
    mask = np.repeat(mask[:, :, np.newaxis], data.shape[2], axis=2)

    col_keys = [dicionary["feature"] for dicionary in scaler_params]
    print(col_keys)
    # drop Pileup_nTrueInt from preprocess_ops
    preprocess_ops.pop("Pileup_nTrueInt", None)
    for (feature, col) in zip(range(data.shape[2]), preprocess_ops.keys()):
        flat_data = data[:, :, feature].flatten()
        if col in col_keys:
            i = col_keys.index(col)
            col_dict = scaler_params[i]
        else:
            col_dict = None
        print(col)
        print(col_dict)
        for op in preprocess_ops[col]:
            if op == None:
                continue
            elif op == "scale":
                mean = col_dict['mean']
                std = col_dict['std']
                flat_data[mask] = flat_data[mask] * std + mean
            elif op == "smear":
                flat_data[mask] = np.rint(flat_data[mask])
        
        data[:, :, feature] = flat_data.reshape(data.shape[0], data.shape[1])

    return data


def plot_jet_features(data, ref, pad_mask, idx, columns, save_dir):
    """
    Plot the features of the jet n idx (only the non-padded features)
    and compares it with the reference
    """
    # drop Pileup_nTrueInt from columns
    columns.pop(0)
    mask = ~pad_mask
    for feature, col in zip(range(data.shape[2]), columns):
        data_flatten = data[:, idx, feature][mask[:, idx]].flatten()
        ref_flatten = ref[:, idx, feature][mask[:, idx]].flatten()
        ws = wasserstein_distance(data_flatten, ref_flatten)

        fig = plt.figure(figsize=(18, 9))
        gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1])
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1], sharex=ax1)
        ax3 = plt.subplot(gs[2])
        ax4 = plt.subplot(gs[3], sharex=ax3)

        fig.suptitle(f"{col} comparison")

        # if i != 4 or i != 11 or i != 12:
        #     bins = 100
        # elif i == 14:
        #     bins = np.arange(-0.5, 9.5, 1)
        # else:
        #     bins = np.arange(-0.5, 80.5, 1)
        bins = 100

        # Linear scale plot
        _, rangeR, _ = ax1.hist(ref_flatten, histtype="step", lw=1, bins=bins, label="FullSim")

        saturated_samples = np.where(data_flatten < np.min(rangeR), np.min(rangeR), data_flatten)
        saturated_samples = np.where(
            saturated_samples > np.max(rangeR), np.max(rangeR), saturated_samples
        )
        
        ax1.hist(
            saturated_samples,
            histtype="step",
            lw=1,
            range=[np.min(rangeR), np.max(rangeR)],
            bins=bins,
            label=f"FlashSim, ws={round(ws, 4)}",
        )
        ax1.legend(frameon=False, loc="upper right")

        # Ratio plot for linear scale
        hist_reco, bins_reco = np.histogram(
            ref_flatten, bins=bins, range=[np.min(rangeR), np.max(rangeR)]
        )
        hist_flash, bins_flash = np.histogram(
            saturated_samples, bins=bins, range=[np.min(rangeR), np.max(rangeR)]
        )

        ratio = np.where(hist_reco > 0, hist_flash / hist_reco, 0)
        ax3.scatter(bins_reco[:-1], ratio, marker=".", color="b")
        ax3.set_ylabel("Flash/Reco")
        # horizontal line at 1
        ax3.axhline(y=1, color="r", linestyle="--", alpha=0.5)
        ax3.set_ylim(0, 2)

        # Log scale plot
        ax2.set_yscale("log")
        ax2.hist(ref_flatten, histtype="step", lw=1, bins=bins)
        ax2.hist(
            saturated_samples,
            histtype="step",
            lw=1,
            range=[np.min(rangeR), np.max(rangeR)],
            bins=bins,
        )

        # Ratio plot for log scale
        ax4.scatter(bins_reco[:-1], ratio, marker=".", color="b")
        ax4.set_ylabel("Flash/Reco")
        # horizontal line at 1
        ax4.axhline(y=1, color="r", linestyle="--", alpha=0.5)
        ax4.set_ylim(0, 2)

        plt.savefig(os.path.join(save_dir, f"{idx}_{col}.png"))
        plt.savefig(os.path.join(save_dir, f"{idx}{col}.pdf"))
        # if writer is not None:
        #     writer.add_figure(f"{columns[i]}", fig, global_step=epoch)
        #     writer.add_scalar(
        #         f"ws/{columns[i]}_wasserstein_distance", ws, global_step=epoch
        #     )
        plt.close()