'''
Blind Source Separation using Independent Vector Analysis with Auxiliary Function

2018 (c) Robin Scheibler, MIT License
'''
import numpy as np

from pyroomacoustics.bss import projection_back

def blinkiva(X, U, n_src=None, sparse_reg=0., estimate_noise=False,
        n_iter=20, n_nmf_pre_iter=30, n_nmf_sub_iter=4, n_iva_sub_iter=4,
        proj_back=True, W0=None, seed=None,
        return_filters=False, callback=None):

    '''
    Implementation of AuxIVA algorithm for BSS presented in

    N. Ono, *Stable and fast update rules for independent vector analysis based
    on auxiliary function technique*, Proc. IEEE, WASPAA, 2011.

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

    if _ != n_frames:
        raise ValueError('The microphones and blinky signals should have the same number of frames')

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

    I = np.eye(n_chan,n_chan)
    Y = np.zeros((n_frames, n_freq, n_chan), dtype=X.dtype)
    V = np.zeros((n_freq, n_chan, n_chan, n_chan), dtype=X.dtype)

    # initialize the parts of NMF
    R_all = np.ones((n_frames, n_chan))
    R = R_all[:,:n_src]  # subset tied to NMF of blinkies
    '''
    R[:,:] = np.ones((n_frames, n_src))
    #R[:,:] = np.sum(np.abs(X) ** 2, axis=1)
    G = np.ones((n_src, n_blink)) * U.mean(axis=0, keepdims=True) / n_src
    '''
    if seed is not None:
        rng_state = np.random.get_state()
        np.random.seed(seed)

    R[:,:] = 0.1 + 0.9 * np.random.rand(n_frames, n_src)
    G = 0.1 + 0.9 * np.random.rand(n_src, n_blink)
    R *= np.linalg.norm(U) / np.linalg.norm(R)

    if seed is not None:
        np.random.set_state(rng_state)


    if estimate_noise:
        zn = np.ones(n_blink) * np.min(U, axis=0)
    else:
        zn = np.zeros(n_blink)

    '''
    U_hat = np.dot(R, G)
    norm_fact = np.linalg.norm(U_hat)
    R *= np.linalg.norm(U) / np.linalg.norm(U_hat)
    '''

    # Final normalization and demixing
    norm = np.mean(R, axis=0, keepdims=True)
    R /= norm
    G *= norm.T
    W[:,:,:n_src] /= np.sqrt(norm[None,:,:])

    # Compute the demixed output
    def demix(Y, X, W):
        for f in range(n_freq):
            Y[:,f,:] = np.dot(X[:,f,:], np.conj(W[f,:,:]))

    # NMF only
    for epoch in range(n_nmf_pre_iter):

        # Update the activations
        U_hat = np.dot(R, G) + zn[None,:]
        U_hat_I = 1. / U_hat
        R_I = 1. / R
        #R *= np.sqrt( (P * R_I ** 2 + np.dot(U * U_hat_I ** 2, G.T)) / (n_freq * R_I + (n_freq - 1) * np.dot(U_hat_I, G.T)) )
        R *= np.sqrt( (np.dot(U * U_hat_I ** 2, G.T)) / (np.dot(U_hat_I, G.T) + sparse_reg) )
        R[R < machine_epsilon] = machine_epsilon

        # Update the gains
        U_hat = np.dot(R, G) + zn[None,:]
        U_hat_I = 1. / U_hat
        #G *= np.sqrt( np.dot(R.T, U * U_hat_I ** 2) / ((n_freq - 1) * np.dot(R.T, U_hat_I)) )
        G *= np.sqrt( np.dot(R.T, U * U_hat_I ** 2) / (np.dot(R.T, U_hat_I)) )
        G[G < machine_epsilon] = machine_epsilon

        # Update noise gain
        if estimate_noise:
            zn *= np.sqrt( np.sum(U * U_hat_I ** 2, axis=0) / np.sum(U_hat_I, axis=0) )

        # enforce normalization of variables
        norm = np.mean(R, axis=0, keepdims=True)
        R /= norm
        G *= norm.T
        W[:,:,:n_src] /= np.sqrt(norm[None,:,:])


    # NMF + IVA

    # initial demixing
    demix(Y, X, W)

    for epoch in range(n_iter):

        if callback is not None and epoch % 10 == 0:
            if proj_back:
                z = projection_back(Y, X[:,:,0])
                callback(Y * np.conj(z[None,:,:]))
            else:
                callback(Y, extra=[W,G,R,X,U])


        # compute activations for sources not tied to NMF
        R_all[:,n_src:] = np.sqrt(np.sum(np.abs(Y[:,:,n_src:] * np.conj(Y[:,:,n_src:])), axis=1))
        
        # Compute Auxiliary Variable
        # shape: (n_freq, n_src, n_mic, n_mic)
        V = np.mean((X[:,:,None,:,None] / (1e-10 + R_all[:,None,:,None,None])) * np.conj(X[:,:,None,None,:]), axis=0)

        # Update now the demixing matrix
        #for s in range(n_src):
        for s in range(n_chan):

            # only update the demixing matrices where there is signal
            I_up = np.max(V[:,s,:,:], axis=(1,2)) > 1e-10

            W_H = np.conj(np.swapaxes(W, 1, 2))
            WV = np.matmul(W_H[I_up], V[I_up,s,:,:])
            rhs = I[None,:,s][[0] * WV.shape[0],:]
            W[I_up,:,s] = np.linalg.solve(WV, rhs)

            P1 = np.conj(W[I_up,:,s])
            P2 = np.sum(V[I_up,s,:,:] * W[I_up,None,:,s], axis=-1)
            W[I_up,:,s] /= np.sqrt(np.sum(P1 * P2, axis=1))[:,None]

        demix(Y, X, W)

        # The sources powers summed over frequencies
        # shape: (n_frames, n_src)
        P = np.linalg.norm(Y[:,:,:n_src], axis=1)

        # enforce normalization of variables
        norm = 1. / np.sqrt(np.mean(np.abs(Y[:,:,:n_src]), axis=(0,1)))
        P *= norm[None,:]
        R *= norm[None,:]
        G /= norm[:,None]
        W[:,:,:n_src] *= norm[None,None,:]

        if (epoch + 1) % n_iva_sub_iter != 0:
            continue

        for sub_epoch in range(n_nmf_sub_iter):

            # Update the activations
            U_hat = np.dot(R, G) + zn[None,:]
            U_hat_I = 1. / U_hat
            R_I = 1. / R
            #R *= np.sqrt( (P * R_I ** 2 + np.dot(U * U_hat_I ** 2, G.T)) / (n_freq * R_I + (n_freq - 1) * np.dot(U_hat_I, G.T) + sparse_reg) )
            R *= np.sqrt( (P * R_I ** 2 + np.dot(U * U_hat_I ** 2, G.T)) / (R_I + np.dot(U_hat_I, G.T) + sparse_reg) )
            R[R < machine_epsilon] = machine_epsilon

            # Update the gains
            U_hat = np.dot(R, G) + zn[None,:]
            U_hat_I = 1. / U_hat
            #G *= np.sqrt( np.dot(R.T, U * U_hat_I ** 2) / ((n_freq - 1) * np.dot(R.T, U_hat_I)) )
            G *= np.sqrt( np.dot(R.T, U * U_hat_I ** 2) / (np.dot(R.T, U_hat_I)) )
            G[G < machine_epsilon] = machine_epsilon

            # Update noise gain
            if estimate_noise:
                zn *= np.sqrt( np.sum(U * U_hat_I ** 2, axis=0) / np.sum(U_hat_I, axis=0) )


    '''
    norm = np.mean(R, axis=0, keepdims=True)
    R /= norm
    G *= norm.T
    W[:,:,:n_src] /= np.sqrt(norm[None,:,:])
    '''

    if proj_back:
        z = projection_back(Y, X[:,:,0])
        Y *= np.conj(z[None,:,:])

    if return_filters:
        return Y, W, G, R, z
    else:
        return Y
