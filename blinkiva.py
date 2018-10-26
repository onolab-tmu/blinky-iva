'''
Blind Source Separation using Independent Vector Analysis with Auxiliary Function

2018 (c) Robin Scheibler, MIT License
'''
import numpy as np

from pyroomacoustics.bss import projection_back

def blinkiva(X, U, n_src=None, sparse_reg=0.,
        n_iter=20, n_nmf_sub_iter=4,
        proj_back=True, W0=None, R0=None, seed=None, print_cost=False,
        return_filters=False, callback=None):

    '''
    Implementation of BlinkIVA algorithm for BSS presented by

    Robin Scheibler, Nobutaka Ono at ICASSP 2019, title to be determined

    Parameters
    ----------
    X: ndarray (nframes, nfrequencies, nchannels)
        STFT representation of the signal
    U: ndarray (nframes, nblinkies)
        The matrix containing the blinky signals
    n_src: int, optional
        The number of sources or independent components
    n_iter: int, optional
        The number of iterations (default 20)
    proj_back: bool, optional
        Scaling on first mic by back projection (default True)
    W0: ndarray (nfrequencies, nchannels, nchannels), optional
        Initial value for demixing matrix
    return_filters: bool
        If true, the function will return the demixing matrix too
    callback: func
        A callback function called every 10 iterations, allows to monitor convergence

    Returns
    -------
    Returns an (nframes, nfrequencies, nsources) array. Also returns
    the demixing matrix (nfrequencies, nchannels, nsources)
    if ``return_values`` keyword is True.
    '''

    n_frames, n_freq, n_chan = X.shape
    _, n_blink = U.shape

    assert n_frames == _, 'Microphones and blinkies do not have the same number of frames ({} != {})'.format(n_frames, _)

    # default to determined case
    if n_src is None:
        n_src = X.shape[2]

    # initialize the demixing matrices
    if W0 is None:
        W = np.array([np.eye(n_chan, n_chan) for f in range(n_freq)], dtype=X.dtype)
    else:
        W = W0.copy()

    # we will threshold entries of R and G that get to small
    machine_epsilon = np.finfo(float).eps

    # U normalized by number of frequencies is used
    U_mean = U / n_freq / 2

    I = np.eye(n_chan,n_chan)
    Y = np.zeros((n_frames, n_freq, n_chan), dtype=X.dtype)
    V = np.zeros((n_freq, n_chan, n_chan, n_chan), dtype=X.dtype)
    P = np.zeros((n_frames, n_chan))

    if seed is not None:
        rng_state = np.random.get_state()
        np.random.seed(seed)

    # initialize the parts of NMF
    if R0 is None:
        R_all = np.ones((n_frames, n_chan))
        R = R_all[:,:n_src]  # subset tied to NMF of blinkies

        R[:,:] = 0.1 + 0.9 * np.random.rand(n_frames, n_src)
        R *= np.mean(np.abs(X[:,:,:n_src]) ** 2, axis=(0,1)) / R.sum(axis=0)
    else:
        R_all = R0.copy()
        R = R_all[:,:n_src]  # subset tied to NMF of blinkies

    G = 0.1 + 0.9 * np.random.rand(n_src, n_blink)
    G *= np.mean(U_mean) / np.mean(G)

    if seed is not None:
        np.random.set_state(rng_state)

    # Compute the demixed output
    def demix(Y, X, W, P, R_all):
        for f in range(n_freq):
            Y[:,f,:] = np.dot(X[:,f,:], np.conj(W[f,:,:]))

        # The sources magnitudes averaged over frequencies
        # shape: (n_frames, n_src)
        P[:,:] = np.linalg.norm(Y, axis=1)

        # copy activations for sources not tied to NMF
        R_all[:,n_src:] = (P[:,n_src:] / n_freq) ** 2

    def R_update(P, U, R, G):
        # Update the activations
        U_hat = np.dot(R, G)
        U_hat_I = 1. / U_hat
        R_I = 1. / R
        num = (0.5 * (P[:,:n_src] / n_freq) * R_I ** 1.5 + np.dot(U_mean * U_hat_I ** 2, G.T))
        denom = (0.5 * R_I + np.dot(U_hat_I, G.T) + sparse_reg)
        R *= np.sqrt(num  / denom)
        R[R < machine_epsilon] = machine_epsilon

    def G_update(U, R, G):
        U_hat = np.dot(R, G)
        U_hat_I = 1. / U_hat
        num = np.dot(R.T, U_mean * U_hat_I ** 2)
        denom = np.dot(R.T, U_hat_I)
        G *= np.sqrt(num  / denom)
        G[G < machine_epsilon] = machine_epsilon

    def cost_function(Y, U, R, G, W):
        U_hat = np.dot(R[:,:G.shape[0]], G)
        cf = - Y.shape[0] * np.sum(np.linalg.slogdet(W)[1])
        cf += np.sum(0.5 * Y.shape[1] * np.log(R) + np.linalg.norm(Y, axis=1) / R ** 0.5)
        cf += np.sum(Y.shape[1] * np.log(U_hat) + U / U_hat / 2)
        return cf

    cost_func_list = []

    # initial demixing
    demix(Y, X, W, P, R_all)

    # NMF + IVA
    for epoch in range(n_iter):

        if callback is not None and epoch % 10 == 0:
            if proj_back:
                z = projection_back(Y, X[:,:,0])
                callback(Y * np.conj(z[None,:,:]))
            else:
                callback(Y, extra=[W,G,R,X,U])

        if print_cost and epoch % 5 == 0:
            cost_func_list.append(cost_function(Y, U, R_all, G, W))
            print('epoch:', epoch, 'cost function ==', cost_func_list[-1])

        # several subiteration of NMF
        for sub_epoch in range(n_nmf_sub_iter):

            # Update the activations
            R_update(P, U, R, G)

            # Update the gains
            G_update(U, R, G)

            # Rescale all variables before continuing
            lmb = 1. / np.mean(R_all, axis=0)
            R_all *= lmb[None,:]
            W *= np.sqrt(lmb[None,None,:])
            P *= np.sqrt(lmb[None,:])
            G /= lmb[:n_src,None]

        # Compute Auxiliary Variable
        # shape: (n_freq, n_src, n_mic, n_mic)
        denom = 4 * P * R_all ** 0.5  # / n_freq  # when I add this, separation improves a lot!
        #denom = 4 * R_all ** 0.5  # / n_freq  # when I add this, separation improves a lot!
        #denom = 4 * R_all  # / n_freq  # when I add this, separation improves a lot!
        V = np.mean(
                (X[:,:,None,:,None] / (1e-15 + denom[:,None,:,None,None]))
                * np.conj(X[:,:,None,None,:]),
                axis=0,
                )

        
        # Update the demixing matrices
        for s in range(n_chan):

            # only update the demixing matrices where there is signal
            I_up = np.max(V[:,s,:,:], axis=(1,2)) > 1e-15

            W_H = np.conj(np.swapaxes(W, 1, 2))
            WV = np.matmul(W_H[I_up], V[I_up,s,:,:])
            rhs = I[None,:,s][[0] * WV.shape[0],:]
            W[I_up,:,s] = np.linalg.solve(WV, rhs)

            P1 = np.conj(W[I_up,:,s])
            P2 = np.sum(V[I_up,s,:,:] * W[I_up,None,:,s], axis=-1)
            W[I_up,:,s] /= np.sqrt(np.sum(P1 * P2, axis=1))[:,None]

        demix(Y, X, W, P, R_all)

        # Rescale all variables before continuing
        lmb = 1. / np.mean(R_all, axis=0)
        R_all *= lmb[None,:]
        W *= np.sqrt(lmb[None,None,:])
        P *= np.sqrt(lmb[None,:])
        G /= lmb[:n_src,None]

    import matplotlib.pyplot as plt
    plt.plot(R[:,0] * n_freq)
    plt.plot(P[:,0])
    import pdb; pdb.set_trace()

    if proj_back:
        z = projection_back(Y, X[:,:,0])
        Y *= np.conj(z[None,:,:])

    if return_filters:
        return Y, W, G, R_all
    else:
        return Y
