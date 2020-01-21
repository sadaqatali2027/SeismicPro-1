""" Seismic batch tools """
import numpy as np

def _crop(image, coords, shape):
    """ Perform crops from the image.
    Number of crops is defined by the number of elements in `coords` parameter.

    Parameters
    ----------
    image : np.array
        Image to crop from.
    coords: list of tuples
        The list of top-left (x,y) coordinates for each crop.
    shape: tuple of ints
        Crop shape.

    Returns
    -------
    res: np.array, dtype='O'
        Array with crops.
    """
    res = np.empty((len(coords), ), dtype='O')
    for i, (x, y) in enumerate(coords):
        if (x + shape[0] > image.shape[0]) or (y + shape[1] > image.shape[1]):
            print(x, y, shape, image.shape)
            raise ValueError('Resulting crop shape is less than expected.')
        res[i] = image[x:x+shape[0], y:y+shape[1]]
    return res
