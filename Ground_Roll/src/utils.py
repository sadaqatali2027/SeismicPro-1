""" utilities """
from functools import reduce

import numpy as np

from matplotlib import pyplot as plt

from seismicpro.batchflow import Pipeline, V, B

from seismicpro.src import FieldIndex, SeismicDataset, seismic_plot


def make_index(paths, index_type=FieldIndex, extra_headers=None, index_name=None):
    """ make index given components and paths"""
    return reduce(lambda x, y: x.merge(y),
                  (index_type(name=name, path=path, extra_headers=extra_headers, index_name=index_name)
                   for name, path in paths.items()))

def load_arrs(i, index, components, index_type=FieldIndex, index_name=None):
    """ load i-th field from index"""
    ppl = Pipeline()
    ppl = reduce(lambda p, c: p.init_variable(c), components, ppl)

    ppl = ppl.load(components=components, fmt='segy', tslice=np.arange(3000))

    ppl = reduce(lambda p, c: p.sort_traces(src=c, dst=c, sort_by='offset'), components, ppl)
    ppl = reduce(lambda p, c: p.update(V(c), B(c)), components, ppl)

    findex = index_type(index, index_name=index_name)
    test_set = SeismicDataset(findex.create_subset(findex.indices[i: i+1]))

    ppl = ppl << test_set
    ppl = ppl.run(batch_size=1, n_epochs=1, drop_last=False, shuffle=False)

    return {c: ppl.get_variable(c)[0] for c in components}


def check_res(i, index, components, mode='img', index_type=FieldIndex, index_name=None, **kwargs):
    """ load i-th item from index and draw it with :meth:`seismic_plot`"""
    arrs = load_arrs(i, index, components=components, index_type=index_type, index_name=index_name)

    fig_kwargs = {}
    if mode == 'wiggle':
        fig_kwargs.update({'wiggle': True, 'xlim': None, 'ylim': None, 'std': 1, **kwargs})
    else:
        crange = kwargs.pop('cv', 0.1)
        fig_kwargs.update({'cmap': 'gray', 'vmin': -crange, 'vmax': crange, **kwargs})

    seismic_plot(list(arrs.values()), names=list(arrs.keys()), title='Field {}'.format(index.indices[i]), **fig_kwargs)

    return arrs


def visualize_geom(df):
    """
    cols = [
        'TRACE_SEQUENCE_LINE',
        'FieldRecord', # 'FieldRecord' = 'SourceY' x 'SourceX'
        'SourceY', 'SourceX', 'GroupY', 'GroupX',
        'CDP_X', # 'CROSSLINE_3D'
        'CDP_Y', # 'INLINE_3D'
        'CDP', # 'CDP_X' x 'CDP_Y'
        'offset',

        'ReceiverDatumElevation', # rline
        'SourceDatumElevation', # reciever id inside rline
        'SourceWaterDepth', # sline
        'GroupWaterDepth', # source id in sline
    ]

    """
    print("Fields: {}, bins: {}".format(df.FieldRecord.nunique(), df.CDP.nunique()))

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    ax1.scatter(x=df['GroupX'], y=df['GroupY'], c='green', marker='^', label='Reciever')
    ax1.scatter(x=df['CDP_X'], y=df['CDP_Y'], c='blue', marker='.', label='CDP', alpha=0.1)

    df1 = df[['FieldRecord', 'SourceX', 'SourceY']].droplevel(1, axis=1).drop_duplicates().set_index('FieldRecord')
    df1['tr_count'] = df['FieldRecord'].value_counts()
    ref = ax1.scatter(x=df1['SourceX'], y=df1['SourceY'], c=df1['tr_count'], marker='v', cmap='jet')

    plt.colorbar(ref, ax=ax1)
    ax1.set_title('Geometry + Shots: trace counts')

    ax1.legend()

    def heatmap(index_col, x_col, y_col, fillna=None):
        tr_count = df[index_col].value_counts()
        df1 = df[[index_col, x_col, y_col]].droplevel(1, axis=1).drop_duplicates().set_index(index_col)
        df1['tr_count'] = tr_count

        p = df1.pivot(index=x_col, columns=y_col, values='tr_count')

        if fillna is not None:
            rindex = np.arange(df[x_col].min(), df[x_col].max(), np.diff(np.unique(df[x_col].values)).min())
            cindex = np.arange(df[y_col].min(), df[y_col].max(), np.diff(np.unique(df[y_col].values)).min())
            return p.reindex(rindex).reindex(columns=cindex).fillna(fillna)

        return p

    hm_table = heatmap('CDP', 'CDP_X', 'CDP_Y', fillna=0)
    ref = ax2.imshow(hm_table.values.T, origin='lower', aspect='equal', cmap='seismic')
    plt.colorbar(ref, ax=ax2)
    ax2.set_title('Bins')
