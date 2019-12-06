import numpy as np
from scipy import signal
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import patches

from tqdm import tqdm

from .utils import load_arrs



def get_windowed_spectrogram_dists(smgr, smgl, dist_fn='sum_abs',
                                   time_frame_width=100, noverlap=None, window='boxcar'):
    """
    Calculates distances between traces' spectrograms in sliding windows

    Parameters
    ----------
    smgr, smgl : np.array of shape (traces count, timestamps)
        traces to compute spectrograms on

    dist_fn : 'max_abs', 'sum_abs', 'sum_sq' or callable, optional
        function to calculate distance between 2 specrograms for single trace and single time window
        if callable, should accept 2 arrays of shape (traces count, frequencies, segment times)
        and operate on second axis
        Default is 'sum_abs'

    time_frame_width : int, optional
        nperseg for signal.spectrogram
        see ::meth:: scipy.signal.spectrogram

    noverlap : int, optional
        see ::meth:: scipy.signal.spectrogram

    window : str or tuple or array_like, optional
        see ::meth:: scipy.signal.spectrogram

    Returns
    -------
    np.array of shape (traces count, segment times) with distance heatmap
    """
    kwargs = dict(window=window, nperseg=time_frame_width, noverlap=noverlap, mode='complex')
    *_, spgl = signal.spectrogram(smgl, **kwargs)
    *_, spgr = signal.spectrogram(smgr, **kwargs)

    funcs = {
        'max_abs': lambda spgl, spgr: np.abs(spgl - spgr).max(axis=1),
        'sum_abs': lambda spgl, spgr: np.sum(np.abs(spgl - spgr), axis=1),
        'sum_sq': lambda spgl, spgr: np.sum(np.abs(spgl - spgr) ** 2, axis=1)
    }
    a_l = np.abs(spgl) ** 2 * 2
    a_r = np.abs(spgr) ** 2 * 2

    if callable(dist_fn):  # res(sl, sr)
        res_a = dist_fn(a_l, a_r)
    elif dist_fn in funcs:
        res_a = funcs[dist_fn](a_l, a_r)
    else:
        raise NotImplementedError('modes other than max_abs, sum_abs, sum_sq not implemented yet')

    return res_a


def draw_modifications_dist(modifications, traces_frac=0.1, distances='sum_abs',  # pylint: disable=too-many-arguments
                            vmin=None, vmax=None, figsize=(15, 15),
                            time_frame_width=100, noverlap=0, window='boxcar',
                            n_cols=None, fontsize=20, aspect=None,
                            save_to=None):
    """
    Draws seismograms with distances computed relative to 1-st given seismogram

    Parameters
    ----------
    modifications : list of tuples (np.array, str)
        each tuple represents a seismogram and its label
        traces in seismograms should be ordered by absolute offset increasing

    traces_frac : float, optional
        fraction of traces to use to compute metrics

    distances : list of str or callables, or str, or callable, optional
        dist_fn to pass to get_windowed_spectrogram_dists
        if list is given, all corresponding metrics values are computed

    vmin, vmax, figsize :
        parameters to pass to pyplot.imshow

    time_frame_width, noverlap, window :
        parameters to pass to get_windowed_spectrogram_dists

    n_cols : int or None, optional
        If int, resulting plots are arranged in n_cols collumns, and several rows, if needed
        if None, resulting plots are arranged in one row

    fontsize : int
        fontsize to use in Axes.set_title

    aspect : 'equal', 'auto', or None
        aspect to pass to Axes.set_aspect. If None, set_aspect is not called
    """

    x, y = 1, len(modifications)
    if n_cols is not None:
        x, y = int(np.ceil(y / n_cols)), n_cols

    _, axs = plt.subplots(x, y, figsize=figsize)

    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])

    axs = axs.flatten()

    origin, _ = modifications[0]
    n_traces, n_ts = origin.shape
    n_use_traces = int(n_traces*traces_frac)

    if isinstance(distances, str) or callable(distances):
        distances = (distances, )

    for i, (mod, description) in enumerate(modifications):
        distances_strings = []
        for dist_fn in distances:
            dist_a = get_windowed_spectrogram_dists(mod[0:n_use_traces], origin[0:n_use_traces],
                                                    dist_fn=dist_fn, time_frame_width=time_frame_width,
                                                    noverlap=noverlap, window=window)

            distances_strings.append(r"$\mu$={:.4}".format(np.mean(dist_a)))

        axs[i].imshow(mod.T, vmin=vmin, vmax=vmax, cmap='gray')
        rect = patches.Rectangle((0, 0), n_use_traces, n_ts, edgecolor='r', facecolor='none', lw=1)
        axs[i].add_patch(rect)
        axs[i].set_title("{},\n{}".format(description, '\n'.join(distances_strings)),
                         fontsize=fontsize)
        if aspect:
            axs[i].set_aspect(aspect)

    if save_to:
        plt.savefig(save_to, transparent=True)

    plt.show()

def validate_all(index, components, base_comp='lift', traces_frac=0.1, distance='sum_abs',
                 time_frame_width=100, noverlap=None, window='boxcar'):
    """ get metrics for all fields in batch """
    res = []
    if base_comp not in components:
        print("Warning! {} not in components provided".format(base_comp))

    for i in tqdm(range(len(index))):
        res.append({})
        
        arrs = load_arrs(i, index, components=components)
        if base_comp in arrs:
            base_comp_arr = arrs.pop(base_comp)
            modifications = [(base_comp_arr, base_comp)]
        else:
            modifications = []
            
        
        modifications += [(arrs[t], t) for t in sorted(arrs)]        

        origin, _ = modifications[0]
        n_traces, _ = origin.shape
        n_use_traces = int(n_traces*traces_frac) if traces_frac <= 1 else min(int(traces_frac), n_traces)

        for mod, description in modifications[1:]:
            dist_a = get_windowed_spectrogram_dists(mod[0:n_use_traces], origin[0:n_use_traces],
                                                    dist_fn=distance, time_frame_width=time_frame_width,
                                                    noverlap=noverlap, window=window)
            res[i][description] = np.mean(dist_a)

    return pd.DataFrame(res).set_index(index.indices)