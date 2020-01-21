""" Utilities for metrics study and validation"""
import numpy as np
from scipy import signal

import matplotlib.pyplot as plt
from matplotlib import patches


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

            distances_strings.append(r"$\mu$={:.4f}".format(np.mean(dist_a)))

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


def spectrum_plot_with_metrics(arrs, frame, rate, max_freq=None, names=None,
                               figsize=None, save_to=None, **kwargs):
    """
    Plot seismogram(s) and power spectrum of given region in the seismogram(s)
    and show distances computed relative to 1-st given seismogram

    Parameters
    ----------
    arrs : array-like
        Seismogram or sequence of seismograms.
    frame : tuple
        List of slices that frame region of interest.
    rate : scalar
        Sampling rate.
    max_freq : scalar
        Upper frequence limit.
    names : str or array-like, optional
        Title names to identify subplots.
    figsize : array-like, optional
        Output plot size.
    save_to : str or None, optional
        If not None, save plot to given path.
    kwargs : dict
        Named argumets to matplotlib.pyplot.imshow.

    """

    if isinstance(arrs, np.ndarray) and arrs.ndim == 2:
        arrs = (arrs,)

    if isinstance(names, str):
        names = (names,)

    origin = arrs[0]
    n_use_traces = frame[0].stop - frame[0].start

    _, ax = plt.subplots(2, len(arrs), figsize=figsize, squeeze=False)
    for i, arr in enumerate(arrs):
        ax[0, i].imshow(arr.T, **kwargs)
        rect = patches.Rectangle((frame[0].start, frame[1].start),
                                 frame[0].stop - frame[0].start,
                                 frame[1].stop - frame[1].start,
                                 edgecolor='r', facecolor='none', lw=2)
        ax[0, i].add_patch(rect)

        dist_m = get_windowed_spectrogram_dists(arr[0:n_use_traces], origin[0:n_use_traces])
        dist = np.mean(dist_m)

        ax[0, i].set_title(r'Seismogram {}. $\mu$={:.4f}'.format(names[i] if names is not None else '', dist))
        ax[0, i].set_aspect('auto')
        spec = abs(np.fft.rfft(arr[frame], axis=1))**2
        freqs = np.fft.rfftfreq(len(arr[frame][0]), d=rate)
        if max_freq is None:
            max_freq = np.inf

        mask = freqs <= max_freq
        ax[1, i].plot(freqs[mask], np.mean(spec, axis=0)[mask], lw=2)
        ax[1, i].set_xlabel('Hz')
        ax[1, i].set_title('Spectrum plot {}'.format(names[i] if names is not None else ''))
        ax[1, i].set_aspect('auto')

        if save_to:
            plt.savefig(save_to, transparent=True)

        plt.show()


def get_modifications_list(batch, i):
    """ get seismic batch components with short names """
    res = []
    # lift should always be the first component
    if 'lift' in batch.components:
        res.append((batch.__getattr__('lift')[i], 'LIFT'))

    res += [(batch.__getattr__(c)[i], c.upper()) for c in batch.components if c != 'lift']

    return res


def validate_all(batch, traces_frac=0.1, distance='sum_abs',
                 time_frame_width=100, noverlap=None, window='boxcar'):
    """ get metrics for all fields in batch """
    res = []

    for i in range(len(batch.index)):
        res.append({})

        modifications = get_modifications_list(batch, i)

        origin, _ = modifications[0]
        n_traces, _ = origin.shape
        n_use_traces = int(n_traces*traces_frac)

        for mod, description in modifications:
            dist_a = get_windowed_spectrogram_dists(mod[0:n_use_traces], origin[0:n_use_traces],
                                                    dist_fn=distance, time_frame_width=time_frame_width,
                                                    noverlap=noverlap, window=window)
            res[i][description + '_amp'] = np.mean(dist_a)

    return res


def get_cv(arrs, q=0.95):
    """
    Calculates upper border for data range covered by a colormap in pyplot.imshow
    """
    return np.abs(np.quantile(np.stack(item for item in arrs), q))
