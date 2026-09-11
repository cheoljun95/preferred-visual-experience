import numpy as np

def lanczosinterp2D(data, oldtime, newtime, window=3):
    """Interpolates the columns of [data], assuming that the i'th row of data corresponds to
    oldtime(i).  A new matrix with the same number of columns and a number of rows given
    by the length of [newtime] is returned.  If [causal], only past time points will be used
    to computed the present value, and future time points will be ignored.
    
    The time points in [newtime] are assumed to be evenly spaced, and their frequency will
    be used to calculate the low-pass cutoff of the sinc interpolation filter.
    
    [window] lobes of the sinc function will be used.  [window] should be an integer.
    """
    ## Find the cutoff frequency ##
    cutoff = 1/np.mean(np.diff(newtime))
    
    ## Construct new signal ##
    # newdata = np.zeros((len(newtime), data.shape[1]))
    # for ndi in range(len(newtime)):
    #         for di in range(len(oldtime)):
    #             newdata[ndi,:] += sincfun(cutoff, newtime[ndi]-oldtime[di], window, causal) * data[di,:]
    
    ## Build up sinc matrix ##
    sincmat = np.zeros((len(newtime), len(oldtime)))
    for ndi in range(len(newtime)):
        sincmat[ndi,:] = lanczosfun(cutoff, newtime[ndi]-oldtime, window)
    ## Construct new signal by multiplying the sinc matrix by the data ##
    newdata = np.dot(sincmat, data)

    return newdata

def lanczosfun(cutoff, t, window=3):
    """Compute the lanczos function with some cutoff frequency [B] at some time [t].
    [t] can be a scalar or any shaped numpy array.
    If given a [window], only the lowest-order [window] lobes of the sinc function
    will be non-zero.
    """
    t = t * cutoff
    val = window * np.sin(np.pi*t) * np.sin(np.pi*t/window) / (np.pi**2 * t**2) *cutoff
    val[t==0] = 1.0
    val[np.abs(t)>window] = 0.0
    return val
