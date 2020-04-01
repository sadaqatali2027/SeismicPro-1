"file with helpful functions"
import numpy as np

from numba import njit, prange


RGB_CONST = np.array([[0, 0, 255],
                      [1, 171, 255],
                      [1, 212, 255],
                      [1, 255, 255],
                      [1, 255, 213],
                      [0, 255, 171],
                      [0, 255, 1],
                      [171, 255, 1],
                      [213, 255, 0],
                      [213, 255, 0],
                      [255, 255, 1],
                      [255, 213, 1],
                      [255, 171, 0],
                      [255, 0, 0],
                      [255, 0, 171],
                      [253, 0, 213]])/255


def running_mean(item, window_width):
    """Calculate running mean with given window."""
    cumsum = np.cumsum(np.insert(item, 0, 0))
    return (cumsum[window_width:] - cumsum[:-window_width])/window_width

@njit
def _calc_semb_easy(field, velocity, t_zero, t_step, offset, semblance):
    for j, vel in enumerate(velocity):
        for k, t_z in enumerate(t_zero):
            sem_val = 0.
            t_sq = t_z**2
            for i, off in enumerate(offset):
                new_t = np.round(np.sqrt(t_sq + (off/vel)**2))
                new_t /= t_step
                if new_t > field.shape[1]-1:
                    break
                new_ix = np.int32(new_t)
                sem_val += field[i][new_ix]
            semblance[j, k] = sem_val
    return semblance

@njit
def _calc_semb_hard(field, velocity, t_zero, t_step, offset, semblance, middle):
    num = np.zeros_like(semblance)
    den = np.zeros_like(semblance)
    max_val = int(t_zero[-1]/t_step)-1

    for ix_t in prange(field.shape[0]):
        trace = field[ix_t]
        for vi in prange(len(velocity)):
            vel = velocity[vi]
            for itera in prange(len(t_zero)):
                time = t_zero[itera]
                ti = np.sqrt(time**2 + offset[ix_t]**2/vel**2)
                ti /= t_step
                iti = np.int32(ti)
                if iti > field.shape[1]-1:
                    break
                num[vi][itera] += trace[iti]
                den[vi][itera] += trace[iti]**2

    t_zero = t_zero + middle
    t_zero = t_zero[:-middle]
    for vi in prange(len(velocity)):
        vel = velocity[vi]
        for itera in prange(len(t_zero)):
            time = t_zero[itera]
            time /= t_step
            time = np.int32(time)
            ismin = time - middle if time - middle > 0 else 0
            ismax = time + middle if time + middle < max_val else max_val

            nsum = dsum = 0
            for i in range(ismin, ismax):
                nsum += num[vi][i]**2
                dsum += den[vi][i]
            semblance[vi][itera] = nsum/(field.shape[0]*dsum + 1e-6) if dsum >= 1e-8 else 0
    return semblance

@njit
def _calc_semb_hard_numba_mx(field, velocity, t_zero, t_step, offset, semblance, middle):
    num = np.zeros_like(semblance)
    den = np.zeros_like(semblance)
    max_val = int(t_zero[-1]/t_step)-1

    time = np.reshape(t_zero, (-1, 1))
    offset_rep = np.zeros((time.shape[0], len(offset)))
    for i in prange(time.shape[0]):
        offset_rep[i] = offset
    for vi in prange(len(velocity)):
        vel = velocity[vi]
        t_new = np.sqrt(time**2 + (offset_rep/vel)**2)
        t_new /= t_step
        int_t_new = t_new.astype(np.int32)
        for i in prange(int_t_new.shape[1]):
            for j in prange(time.shape[0]):
                ix = int_t_new[j][i]
                if ix > field.shape[1] - 1:
                    break
                num[vi][j] += field[i, ix]
                den[vi][j] += (field[i, ix])**2


    t_zero = t_zero + middle
    t_zero = t_zero[:-middle]
    for vi in prange(len(velocity)):
        vel = velocity[vi]
        for itera in prange(len(t_zero)):
            ttime = t_zero[itera]
            ttime /= t_step
            ttime = np.int32(ttime)
            ismin = ttime - middle if ttime - middle > 0 else 0
            ismax = ttime + middle if ttime + middle < max_val else max_val

            nsum = dsum = 0
            for i in range(ismin, ismax):
                nsum += num[vi][i]**2
                dsum += den[vi][i]
            semblance[vi][itera] = nsum/(field.shape[0]*dsum + 1e-6) if dsum >= 1e-8 else 0
    return semblance


def _calc_semb_hard_matrix(field, velocity, t_zero, t_step, offset, semblance, middle):
    num = np.zeros_like(semblance)
    den = np.zeros_like(semblance)

    time = np.reshape(t_zero, (-1, 1))
    offset = offset.reshape(1, *offset.shape)
    offset = np.repeat(offset, time.shape[0], axis=0)
    ix_f = tuple(np.repeat(tuple(np.arange(0, offset.shape[1]).reshape(1, -1)),
                           t_zero.shape[0], axis=0))
    for vi in prange(len(velocity)):
        vel = velocity[vi]
        t_new = np.sqrt(time**2 + (offset/vel)**2)
        t_new /= t_step
        mask = t_new < field.shape[1]
        int_t_new = t_new.astype(np.int32)
        int_t_new[int_t_new >= field.shape[1]] = 0
        vals = field[ix_f, int_t_new] * mask
        num[vi] = np.sum(vals, axis=1)
        den[vi] = np.sum(vals**2, axis=1)

    border = np.arange(len(t_zero) - middle*2 + 1)
    rep_time = np.repeat(np.arange(0, middle*2).reshape(1, -1), border.shape[0], axis=0)
    ixs = rep_time + border.reshape(-1, 1)

    for i in prange(len(ixs)):
        ix = ixs[i]
        nsum = np.sum(num[:, ix]**2, axis=1)
        dsum = np.sum(den[:, ix], axis=1)
        zeros = np.where(dsum < 1e-8)[0]
        result = nsum/(field.shape[0]*dsum + 1e-6)
        result[zeros] = 0
        semblance[:, i] += result
    return semblance
