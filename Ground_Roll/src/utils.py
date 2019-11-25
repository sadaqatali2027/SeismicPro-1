""" utilities """
from functools import reduce

import numpy as np

from seismicpro.batchflow import Pipeline, V, B

from seismicpro.src import FieldIndex, SeismicDataset

def make_index(paths, index_type=FieldIndex, extra_headers=None):
    """ make index given components and paths"""
    return reduce(lambda x, y: x.merge(y), 
                  (index_type(name=name, path=path, extra_headers=extra_headers) for name, path in paths.items()))

def load_arrs(i, index, components):
    """ load i-th field from index"""
    ppl = Pipeline()
    ppl = reduce(lambda p, c: p.init_variable(c), components, ppl)
    ppl = (ppl
           .load(components=components, fmt='segy', tslice=np.arange(3000))
           .sort_traces(src=components, dst=components, sort_by='offset'))
    ppl = reduce(lambda p, c: p.update(V(c), B(c)), components, ppl)
    
    findex = FieldIndex(index)
    test_set = SeismicDataset(findex.create_subset(findex.indices[i: i+1]))

    ppl = ppl << test_set
    ppl = ppl.run(batch_size=1, n_epochs=1, drop_last=False, shuffle=False)

    return {c: ppl.get_variable(c)[0] for c in components}