"""Seismic batch."""
import os
from textwrap import dedent
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import pywt
import segyio

from ..batchflow import action, inbatch_parallel, Batch

from .seismic_index import SegyFilesIndex
from .batch_tools import FILE_DEPENDEND_COLUMNS
from .utils import IndexTracker, partialmethod, write_segy_file

PICKS_FILE_HEADERS = ['FieldRecord', 'TraceNumber', 'ShotPoint', 'timeOffset']


ACTIONS_DICT = {
    "clip": (np.clip, "numpy.clip", "clip values"),
    "gradient": (np.gradient, "numpy.gradient", "gradient"),
    "fft2": (np.fft.fft2, "numpy.fft.fft2", "a Discrete 2D Fourier Transform"),
    "ifft2": (np.fft.ifft2, "numpy.fft.ifft2", "an inverse Discrete 2D Fourier Transform"),
    "fft": (np.fft.fft, "numpy.fft.fft", "a Discrete Fourier Transform"),
    "ifft": (np.fft.ifft, "numpy.fft.ifft", "an inverse Discrete Fourier Transform"),
    "rfft": (np.fft.rfft, "numpy.fft.rfft", "a real-input Discrete Fourier Transform"),
    "irfft": (np.fft.irfft, "numpy.fft.irfft", "a real-input inverse Discrete Fourier Transform"),
    "dwt": (pywt.dwt, "pywt.dwt", "a single level Discrete Wavelet Transform"),
    "idwt": (lambda x, *args, **kwargs: pywt.idwt(*x, *args, **kwargs), "pywt.idwt",
             "a single level inverse Discrete Wavelet Transform"),
    "wavedec": (pywt.wavedec, "pywt.wavedec", "a multilevel 1D Discrete Wavelet Transform"),
    "waverec": (lambda x, *args, **kwargs: pywt.waverec(list(x), *args, **kwargs), "pywt.waverec",
                "a multilevel 1D Inverse Discrete Wavelet Transform"),
    "pdwt": (lambda x, part, *args, **kwargs: pywt.downcoef(part, x, *args, **kwargs), "pywt.downcoef",
             "a partial Discrete Wavelet Transform data decomposition"),
    "cwt": (lambda x, *args, **kwargs: pywt.cwt(x, *args, **kwargs)[0].T, "pywt.cwt", "a Continuous Wavelet Transform"),
}


TEMPLATE_DOCSTRING = """
    Compute {description} for each trace.
    This method simply wraps ``apply_along_axis`` method by setting the
    ``func`` argument to ``{full_name}``.

    Parameters
    ----------
    src : str, optional
        Batch component to get the data from.
    dst : str, optional
        Batch component to put the result in.
    args : misc
        Any additional positional arguments to ``{full_name}``.
    kwargs : misc
        Any additional named arguments to ``{full_name}``.

    Returns
    -------
    batch : SeismicBatch
        Transformed batch. Changes ``dst`` component.
"""
TEMPLATE_DOCSTRING = dedent(TEMPLATE_DOCSTRING).strip()

def apply_to_each_component(method):
    """Combine list of src items and list dst items into pairs of src and dst items
    and apply the method to each pair.

    Parameters
    ----------
    method : callable
        Method to be decorated.

    Returns
    -------
    decorator : callable
        Decorated method.
    """
    def decorator(self, *args, src, dst, **kwargs):
        """Returned decorator."""
        if isinstance(src, str):
            src = (src, )

        if isinstance(dst, str):
            dst = (dst, )

        for isrc, idst in list(zip(src, dst)):
            method(self, *args, src=isrc, dst=idst, **kwargs)

        return self
    return decorator

def add_actions(actions_dict, template_docstring):
    """Add new actions in ``SeismicBatch`` by setting ``func`` argument in
    ``SeismicBatch.apply_to_each_trace`` method to given callables.

    Parameters
    ----------
    actions_dict : dict
        A dictionary, containing new methods' names as keys and a callable,
        its full name and description for each method as values.
    template_docstring : str
        A string, that will be formatted for each new method from
        ``actions_dict`` using ``full_name`` and ``description`` parameters
        and assigned to its ``__doc__`` attribute.

    Returns
    -------
    decorator : callable
        Class decorator.
    """
    def decorator(cls):
        """Returned decorator."""
        for method_name, (func, full_name, description) in actions_dict.items():
            docstring = template_docstring.format(full_name=full_name, description=description)
            method = partialmethod(cls.apply_along_axis, func)
            method.__doc__ = docstring
            setattr(cls, method_name, method)

        return cls
    return decorator


@add_actions(ACTIONS_DICT, TEMPLATE_DOCSTRING)  # pylint: disable=too-many-public-methods,too-many-instance-attributes
class SeismicBatch(Batch):
    """Batch class for seimsic data. Contains seismic traces, metadata and processing methods.

    Parameters
    ----------
    index : DataFrameIndex
        Unique identifiers for sets of seismic traces.
    preloaded : tuple, optional
        Data to put in the batch if given. Defaults to ``None``.

    Attributes
    ----------
    index : DataFrameIndex
        Unique identifiers for sets of seismic traces.
    meta : dict
        Metadata about batch components.
    """
    def __init__(self, index, preloaded=None):
        super().__init__(index, preloaded=preloaded)
        if preloaded is None:
            self.meta = dict()

    def _init_component(self, *args, dst, **kwargs):
        """Create and preallocate a new attribute with the name ``dst`` if it
        does not exist and return batch indices."""
        _ = args, kwargs
        if isinstance(dst, str):
            dst = (dst,)

        for comp in dst:
            if not hasattr(self, comp):
                setattr(self, comp, np.array([None] * len(self.index)))

            if comp not in self.meta:
                self.meta[comp] = dict()

        return self.indices

    @action
    @inbatch_parallel(init="_init_component", target="threads")
    @apply_to_each_component
    def apply_along_axis(self, index, func, *args, src, dst, slice_axis=0, **kwargs):
        """Apply function along specified axis of batch items.

        Parameters
        ----------
        func : callable
            A function to apply. Must accept a trace as its first argument.
        src : str, array-like
            Batch component name to get the data from.
        dst : str, array-like
            Batch component name to put the result in.
        item_axis : int, default: 0
            Batch item axis to apply ``func`` along.
        slice_axis : int
            Axis to iterate data over.
        args : misc
            Any additional positional arguments to ``func``.
        kwargs : misc
            Any additional named arguments to ``func``.

        Returns
        -------
        batch : SeismicBatch
            Transformed batch. Changes ``dst`` component.
        """
        i = self.get_pos(None, src, index)
        src_data = getattr(self, src)[i]
        dst_data = np.array([func(x, *args, **kwargs) for x in np.rollaxis(src_data, slice_axis)])
        getattr(self, dst)[i] = dst_data

    @action
    @apply_to_each_component
    def apply_transform(self, func, *args, src, dst, **kwargs):
        """Apply a function to each item in the batch.

        Parameters
        ----------
        func : callable
            A function to apply. Must accept an item of ``src`` as its first argument.
        src : str, array-like
            The source to get the data from.
        dst : str, array-like
            The source to put the result in.
        args : misc
            Any additional positional arguments to ``func``.
        kwargs : misc
            Any additional named arguments to ``func``.

        Returns
        -------
        batch : SeismicBatch
            Transformed batch.
        """
        super().apply_transform(func, *args, src=src, dst=dst, **kwargs)
        dst_data = getattr(self, dst)
        setattr(self, dst, np.array([i for i in dst_data] + [None])[:-1])
        return self


    @action
    @inbatch_parallel(init="_init_component", target="threads")
    @apply_to_each_component
    def band_pass_filter(self, index, *args, src, dst, lowcut=None, highcut=None, fs=1, order=5):
        """Apply a band pass filter.

        Parameters
        ----------
        src : str, array-like
            The batch components to get the data from.
        dst : str, array-like
            The batch components to put the result in.
        lowcut : real, optional
            Lowcut frequency.
        highcut : real, optional
            Highcut frequency.
        order : int
            The order of the filter.
        fs : real
            Sampling rate.

        Returns
        -------
        batch : SeismicBatch
            Batch with filtered traces.
        """
        _ = args
        i = self.get_pos(None, src, index)
        traces = getattr(self, src)[i]
        nyq = 0.5 * fs
        if lowcut is None:
            b, a = signal.butter(order, highcut / nyq, btype='high')
        elif highcut is None:
            b, a = signal.butter(order, lowcut / nyq, btype='low')
        else:
            b, a = signal.butter(order, [lowcut / nyq, highcut / nyq], btype='band')

        getattr(self, dst)[i] = signal.lfilter(b, a, traces)

    @action
    @inbatch_parallel(init="_init_component", target="threads")
    @apply_to_each_component
    def to_2d(self, index, *args, src, dst, length_alignment=None, pad_value=0):
        """Convert array of 1d arrays to 2d array.

        Parameters
        ----------
        src : str, array-like
            The batch components to get the data from.
        dst : str, array-like
            The batch components to put the result in.
        length_alignment : str, optional
            Defines what to do with arrays of diffetent lengths.
            If 'min', cut the end by minimal array length.
            If 'max', pad the end to maximal array length.
            If None, try to put array to 2d array as is.

        Returns
        -------
        batch : SeismicBatch
            Batch with items converted to 2d arrays.
        """
        _ = args
        pos = self.get_pos(None, src, index)
        data = getattr(self, src)[pos]
        if data is None or len(data) == 0:
            return

        try:
            data_2d = np.vstack(data)
        except ValueError as err:
            if length_alignment is None:
                raise ValueError(str(err) + '\nTry to set length_alingment to \'max\' or \'min\'')
            elif length_alignment == 'min':
                nsamples = min([len(t) for t in data])
            elif length_alignment == 'max':
                nsamples = max([len(t) for t in data])
            else:
                raise NotImplementedError('Unknown length_alingment')
            shape = (len(data), nsamples)
            data_2d = np.full(shape, pad_value)
            for i, arr in enumerate(data):
                data_2d[i, :len(arr)] = arr[:nsamples]

        getattr(self, dst)[pos] = data_2d

    @action
    def dump(self, src, fmt, path, **kwargs):
        """Export data to file.

        Parameters
        ----------
        src : str
            Batch component to dump data from.
        fmt : str
            Output data format.

        Returns
        -------
        batch : SeismicBatch
            Unchanged batch.
        """
        if fmt.lower() in ['sgy', 'segy']:
            return self._dump_segy(src, path, **kwargs)
        if fmt == 'picks':
            return self._dump_picking(src, path, **kwargs)
        raise NotImplementedError('Unknown format.')

    @action
    def _dump_segy(self, src, path, split=True):
        """Dump data to segy files.

        Parameters
        ----------
        path : str
            Path for output files.
        src : str
            Batch component to dump data from.
        split : bool
            Whether to dump batch items into separate files.

        Returns
        -------
        batch : SeismicBatch
            Unchanged batch.
        """
        if split:
            return self._dump_split_segy(src, path)

        return self._dump_single_segy(src, path)

    @inbatch_parallel(init="indices", target="threads")
    def _dump_split_segy(self, index, src, path):
        """Dump data to segy files."""
        pos = self.get_pos(None, src, index)
        data = np.atleast_2d(getattr(self, src)[pos])

        path = os.path.join(path, str(index) + '.sgy')

        df = self.index._idf.loc[[index]] # pylint: disable=protected-access
        sort_by = self.meta[src]['sorting']
        if sort_by is not None:
            df = df.sort_values(by=sort_by)

        df.reset_index(drop=self.index.name is None, inplace=True)
        headers = list(set(df.columns.levels[0]) - set(FILE_DEPENDEND_COLUMNS))
        segy_headers = [h for h in headers if hasattr(segyio.TraceField, h)]
        df = df[segy_headers]
        df.columns = df.columns.droplevel(1)

        write_segy_file(data, df, self.meta[src]['samples'], path)

    def _dump_single_segy(self, src, path):
        """Dump data to segy file."""
        data = np.vstack(getattr(self, src))

        df = self.index._idf # pylint: disable=protected-access
        sort_by = self.meta[src]['sorting']
        if sort_by is not None:
            df = df.sort_values(by=sort_by)

        df = df.loc[self.indices]
        df.reset_index(drop=self.index.name is None, inplace=True)
        headers = list(set(df.columns.levels[0]) - set(FILE_DEPENDEND_COLUMNS))
        segy_headers = [h for h in headers if hasattr(segyio.TraceField, h)]
        df = df[segy_headers]
        df.columns = df.columns.droplevel(1)

        write_segy_file(data, df, self.meta[src]['samples'], path)

        return self

    @action
    def _dump_picking(self, src, path, to_samples=None, columns=None):
        """Dump picking to file.

        Parameters
        ----------
        src : str
            Source to get picking from.
        path : str
            Output file path.
        to_samples : str or scalar or array-like, default to None
            Convertion of the source data interpreted as array of indices to time samples.
            If string, get time samples from corresponding batch component.
            If scalar, the value is interpreted as sampling rate.
            If array-like, the array is interpreted as time samples.
        columns: array_like, optional
            Columns to include in the output file. See PICKS_FILE_HEADERS for default format.

        Returns
        -------
        batch : SeismicBatch
            Batch unchanged.
        """
        data = np.vstack(getattr(self, src)).ravel()
        if to_samples is not None:
            if isinstance(to_samples, str):
                data = self.meta[to_samples]['samples'][data]
            elif len(np.atleast_1d(to_samples)) == 1:
                data = to_samples * data
            else:
                data = to_samples[data]

        if columns is None:
            columns = PICKS_FILE_HEADERS

        df = self.index._idf # pylint: disable=protected-access
        sort_by = self.meta[src]['sorting']
        if sort_by is not None:
            df = df.sort_values(by=sort_by)

        df = df.loc[self.indices]
        df['timeOffset'] = data
        df = df.reset_index(drop=self.index.name is None)[columns]
        df.columns = df.columns.droplevel(1)
        df.to_csv(path, index=False)
        return self

    @action
    def load(self, src=None, fmt=None, components=None, **kwargs):
        """Load data into components.

        Parameters
        ----------
        src : misc, optional
            Source to load components from.
        fmt : str, optional
            Source format.
        components : str or array-like, optional
            Components to load.
        **kwargs: dict
            Any kwargs to be passed to load method.

        Returns
        -------
        batch : SeismicBatch
            Batch with loaded components.
        """
        if fmt.lower() in ['sgy', 'segy']:
            return self._load_segy(src=components, dst=components, **kwargs)
        if fmt == 'picks':
            return self._load_picking(src=src, components=components)

        return super().load(src=src, fmt=fmt, components=components, **kwargs)

    def _load_picking(self, src, components):
        """Load picking from file."""
        df = pd.read_csv(src)
        df.columns = pd.MultiIndex.from_arrays([df.columns, [''] * len(df.columns)])
        idf = self.index._idf # pylint: disable=protected-access
        if self.index.name is not None:
            idf = idf.reset_index()

        df = idf.merge(df, how='left')
        if self.index.name is not None:
            df = df.set_index(self.index.name)

        res = [df.loc[i, 'timeOffset'].values for i in self.indices]
        setattr(self, components, res)
        return self

    @apply_to_each_component
    def _load_segy(self, src, dst, tslice=None):
        """Load data from segy files.

        Parameters
        ----------
        src : str, array-like
            Component to load.
        dst : str, array-like
            The batch component to put loaded data in.
        tslice: slice, optional
            Load a trace subset given by slice.

        Returns
        -------
        batch : SeismicBatch
            Batch with loaded components.
        """
        segy_index = SegyFilesIndex(self.index, name=src)
        idf = segy_index._idf # pylint: disable=protected-access
        order = np.hstack([np.where(idf.index == i)[0] for i in segy_index.indices])

        batch = type(self)(segy_index)._load_from_segy_file(src=src, dst=dst, tslice=tslice) # pylint: disable=protected-access
        all_traces = np.concatenate(getattr(batch, dst))[np.argsort(order)]
        self.meta[dst] = dict(samples=batch.meta[dst]['samples'])

        idf = self.index._idf # pylint: disable=protected-access
        if idf.index.name is None:
            items = [self.get_pos(None, "indices", i) for i in idf.index]
            res = np.array(list(all_traces[items]) + [None])[:-1]
        else:
            res = np.array([None] * len(self))
            for i in self.indices:
                ipos = self.get_pos(None, "indices", i)
                items = np.where(idf.index == i)[0]
                res[ipos] = all_traces[items]

        setattr(self, dst, res)
        self.meta[dst]['sorting'] = None

        return self

    @inbatch_parallel(init="_init_component", target="threads")
    def _load_from_segy_file(self, index, *args, src, dst, tslice=None):
        """Load from a single segy file."""
        _ = src, args
        pos = self.get_pos(None, "indices", index)
        path = index
        trace_seq = self.index._idf.loc[index][('TRACE_SEQUENCE_FILE', src)] # pylint: disable=protected-access
        if tslice is None:
            tslice = slice(None)

        with segyio.open(path, strict=False) as segyfile:
            traces = np.atleast_2d([segyfile.trace[i - 1][tslice] for i in
                                    np.atleast_1d(trace_seq).astype(int)])
            samples = segyfile.samples[tslice]

        getattr(self, dst)[pos] = traces
        if index == self.indices[0]:
            self.meta[dst]['samples'] = samples
            self.meta[dst]['sorting'] = None

        return self

    @action
    @inbatch_parallel(init="_init_component", target="threads")
    @apply_to_each_component
    def slice_traces(self, index, *args, src, dst, slice_obj):
        """
        Slice traces.

        Parameters
        ----------
        src : str, array-like
            The batch components to get the data from.
        dst : str, array-like
            The batch components to put the result in.
        slice_obj : slice
            Slice to extract from traces.

        Returns
        -------
        batch : SeismicBatch
            Batch with sliced traces.
        """
        _ = args
        pos = self.get_pos(None, src, index)
        data = getattr(self, src)[pos]
        getattr(self, dst)[pos] = data[:, slice_obj]
        return self

    @action
    @inbatch_parallel(init="_init_component", target="threads")
    @apply_to_each_component
    def pad_traces(self, index, *args, src, dst, **kwargs):
        """
        Pad traces with ```numpy.pad```.

        Parameters
        ----------
        src : str, array-like
            The batch components to get the data from.
        dst : str, array-like
            The batch components to put the result in.
        kwargs : dict
            Named arguments to ```numpy.pad```.

        Returns
        -------
        batch : SeismicBatch
            Batch with padded traces.
        """
        _ = args
        pos = self.get_pos(None, src, index)
        data = getattr(self, src)[pos]
        pad_width = kwargs['pad_width']
        if isinstance(pad_width, int):
            pad_width = (pad_width, pad_width)

        kwargs['pad_width'] = [(0, 0)] + [pad_width] + [(0, 0)] * (data.ndim - 2)
        getattr(self, dst)[pos] = np.pad(data, **kwargs)
        return self

    @action
    @inbatch_parallel(init="_init_component", target="threads")
    @apply_to_each_component
    def sort_traces(self, index, *args, src, dst, sort_by):
        """Sort traces.

        Parameters
        ----------
        src : str, array-like
            The batch components to get the data from.
        dst : str, array-like
            The batch components to put the result in.
        sort_by: str
            Sorting key.

        Returns
        -------
        batch : SeismicBatch
            Batch with new trace sorting.
        """
        _ = args
        pos = self.get_pos(None, src, index)
        df = self.index._idf.loc[[index]] # pylint: disable=protected-access
        order = np.argsort(df[sort_by].tolist())
        getattr(self, dst)[pos] = getattr(self, src)[pos][order]
        if pos == 0:
            self.meta[dst]['sorting'] = sort_by

        return self

    def items_viewer(self, src, scroll_step=1, **kwargs):
        """Scroll and view batch items. Emaple of use:
        ```
        %matplotlib notebook

        fig, tracker = batch.items_viewer('raw', vmin=-cv, vmax=cv, cmap='gray')
        fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
        plt.show()
        ```

        Parameters
        ----------
        src : str
            The batch component with data to show.
        scroll_step : int, default: 1
            Number of batch items scrolled at one time.
        kwargs: dict
            Additional keyword arguments for matplotlib imshow.

        Returns
        -------
        fig, tracker
        """
        fig, ax = plt.subplots(1, 1)
        tracker = IndexTracker(ax, getattr(self, src), self.indices,
                               scroll_step=scroll_step, **kwargs)
        return fig, tracker

    def imshow(self, src, index, figsize=None, save_to=None, dpi=None, **kwargs):
        """Show data on a 2D regular raster.

        Parameters
        ----------
        src : str
            The batch component with data to show.
        index : same type as batch.indices
            Data index to show.
        figsize :  tuple of integers, optional, default: None
            Image figsize as in matplotlib.
        save_to : str, default: None
            Path to save image.
        dpi : int, optional, default: None
            The resolution argument for matplotlib savefig.
        kwargs: dict
            Additional keyword arguments for matplotlib imshow.

        Returns
        -------
        """
        pos = self.get_pos(None, src, index)
        data = getattr(self, src)[pos]
        if figsize is not None:
            plt.figure(figsize=figsize)

        plt.imshow(data.T, **kwargs)
        plt.title(index)
        plt.axis('auto')
        if save_to is not None:
            plt.savefig(save_to, dpi=dpi)

        plt.show()
