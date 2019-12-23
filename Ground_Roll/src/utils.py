""" utilities """
from functools import reduce

import numpy as np

from seismicpro.batchflow import Pipeline, V, B

from seismicpro.src import FieldIndex, SeismicDataset, seismic_plot

def make_index(paths, index_type=FieldIndex, extra_headers=None, index_name=None):
    """ make index given components and paths"""
    return reduce(lambda x, y: x.merge(y), 
                  (index_type(name=name, path=path, extra_headers=extra_headers, index_name=index_name) for name, path in paths.items()))

def load_arrs(i, index, components, index_type=FieldIndex, index_name=None):
    """ load i-th field from index"""
    ppl = Pipeline()
    ppl = reduce(lambda p, c: p.init_variable(c), components, ppl)
    ppl = (ppl
           .load(components=components, fmt='segy', tslice=np.arange(3000))
           .sort_traces(src=components, dst=components, sort_by='offset'))
    ppl = reduce(lambda p, c: p.update(V(c), B(c)), components, ppl)
    
    findex = index_type(index, index_name=index_name)
    test_set = SeismicDataset(findex.create_subset(findex.indices[i: i+1]))

    ppl = ppl << test_set
    ppl = ppl.run(batch_size=1, n_epochs=1, drop_last=False, shuffle=False)

    return {c: ppl.get_variable(c)[0] for c in components}


def check_res(i, index, components, mode='img', index_type=FieldIndex, index_name=None, **kwargs):
    arrs = load_arrs(i, index, components=components, index_type=index_type, index_name=index_name)   
    
    fig_kwargs = {}
    if mode == 'wiggle':
        fig_kwargs.update({'wiggle':True, 'xlim':None, 'ylim':None, 'std':1, **kwargs})
    else:
        cv = kwargs.pop('cv', 0.1)
        fig_kwargs.update({'cmap':'gray', 'vmin':-cv, 'vmax':cv, **kwargs})
        
    seismic_plot(list(arrs.values()), names=list(arrs.keys()), title='Field {}'.format(index.indices[i]), **fig_kwargs)
    
    return arrs

    