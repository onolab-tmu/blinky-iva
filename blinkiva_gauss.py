# Copyright (c) 2019 Robin Scheibler
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
'''
Blind Source Separation using Independent Vector Analysis with Microphones and Blinkies

2019 (c) Robin Scheibler, MIT License
'''
import numpy as np

from pyroomacoustics.bss import projection_back

def blinkiva_gauss(X, U, n_src=None, sparse_reg=0.,
        n_iter=20, n_nmf_sub_iter=4, epsilon=1.,
        proj_back=True, W0=None, R0=None, seed=None,
        print_cost=False,
        return_filters=False, callback=None):

    '''
    Implementation of BlinkIVA algorithm for blind source separation using jointly
    microphones and sound power sensors "blinkies". The algorithm was presented in

    R. Scheibler and N. Ono, *Multi-modal Blind Source Separation with Microphones and Blinkies,*
    Proc. IEEE ICASSP, Brighton, UK, May, 2019.  DOI: 10.1109/ICASSP.2019.8682594
    https://arxiv.org/abs/1904.02334

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

    # we will only work with the blinky signal normalized by frequency
    U_mean = U / n_freq / 2.

    I = np.eye(n_chan,n_chan)
    Y = np.zeros((n_frames, n_freq, n_chan), dtype=X.dtype)
    V = np.zeros((n_freq, n_chan, n_chan, n_chan), dtype=X.dtype)
    P = np.zeros((n_frames, n_chan))

    # initialize the parts of NMF
    R_all = np.ones((n_frames, n_chan))
    R = R_all[:,:n_src]  # subset tied to NMF of blinkies

    if seed is not None:
        rng_state = np.random.get_state()
        np.random.seed(seed)

    if R0 is None:
        R[:,:] = 0.1 + 0.9 * np.random.rand(n_frames, n_src)
        R /= np.mean(R, axis=0, keepdims=True)
    else:
        R[:,:] = R0

    G = 0.1 + 0.9 * np.random.rand(n_src, n_blink)
    U_hat = np.dot(R, G)
    G *= np.sum(U_mean, axis=0, keepdims=True) / np.sum(U_hat, axis=0, keepdims=True)

    if seed is not None:
        np.random.set_state(rng_state)

    def cost(Y, W, R, G):
        pwr = np.linalg.norm(Y, axis=1) ** 2
        cf1 = -2 * Y.shape[0] * np.sum(np.linalg.slogdet(W)[1])
        cf2 = np.sum(Y.shape[1] * np.log(R) + pwr / R)
        U_hat = np.dot(R[:,:n_src], G)
        cf3 = np.sum(np.log(U_hat) + U / U_hat / 2)
        return { 'iva' : cf1 + cf2, 'nmf' : cf2 + cf3, 'blinkiva' : cf1 + cf2 + cf3 }

    def rescale(W, P, R, G):
        # Fix the scale of all variables
        lmb = 1. / np.mean(R, axis=0)
        R *= lmb[None,:]
        P *= lmb[None,:]
        W *= np.sqrt(lmb[None,None,:])
        G /= lmb[:G.shape[0],None]

    # Compute the demixed output
    def demix(Y, X, W, P, R_all):
        for f in range(n_freq):
            Y[:,f,:] = np.dot(X[:,f,:], np.conj(W[f,:,:]))

        # The sources magnitudes averaged over frequencies
        # shape: (n_frames, n_src)
        P[:,:] = np.linalg.norm(Y, axis=1) ** 2 / n_freq

        # copy activations for sources not tied to NMF
        R_all[:,n_src:] = P[:,n_src:]


    # initial demixing
    demix(Y, X, W, P, R_all)

    cost_joint_list = []

    # NMF + IVA joint updates
    for epoch in range(n_iter):

        if callback is not None and epoch % 10 == 0:

            if proj_back:
                z = projection_back(Y, X[:,:,0])
                callback(Y * np.conj(z[None,:,:]))
            else:
                callback(Y, extra=[W,G,R,X,U])

        if print_cost and epoch % 10 == 0:
            print('Cost function: iva={iva:13.0f} nmf={nmf:13.0f} iva+nmf={blinkiva:13.0f}'.format(
                **cost(Y, W, R_all, G)))

        for sub_epoch in range(n_nmf_sub_iter):

            # Update the activations
            U_hat = np.dot(R, G)
            U_hat_I = 1. / U_hat
            R_I = 1. / R
            R *= np.sqrt(
                    (P[:,:n_src] * R_I ** 2 + np.dot(U_mean * U_hat_I ** 2, G.T))
                    / (R_I + np.dot(U_hat_I, G.T) + sparse_reg)
                    )
            R[R < machine_epsilon] = machine_epsilon

            # Update the gains
            U_hat = np.dot(R, G)
            U_hat_I = 1. / U_hat
            G *= np.sqrt( np.dot(R.T, U_mean * U_hat_I ** 2) / np.dot(R.T, U_hat_I) )
            G[G < machine_epsilon] = machine_epsilon

            # normalize
            #rescale(W, P, R_all, G)

        # Compute Auxiliary Variable
        # shape: (n_freq, n_src, n_mic, n_mic)
        denom = 2 * R_all
        denom[denom < epsilon] = epsilon  # regularize this part

        # 1) promote all arrays to (n_frames, n_freq, n_chan, n_chan, n_chan)
        # 2) take the outer product (complex) of the last two dimensions
        #    to get the covariance matrix over the microphones
        # 3) average over the time frames (index 0)
        V = np.mean(
                (X[:,:,None,:,None] / denom[:,None,:,None,None])
                * np.conj(X[:,:,None,None,:]),
                axis=0,
                )

        # Update now the demixing matrix
        #for s in range(n_src):
        for s in range(n_chan):

            W_H = np.conj(np.swapaxes(W, 1, 2))
            WV = np.matmul(W_H, V[:,s,:,:])
            rhs = I[None,:,s][[0] * WV.shape[0],:]
            W[:,:,s] = np.linalg.solve(WV, rhs)

            P1 = np.conj(W[:,:,s])
            P2 = np.sum(V[:,s,:,:] * W[:,None,:,s], axis=-1)
            W[:,:,s] /= np.sqrt(np.sum(P1 * P2, axis=1))[:,None]

        demix(Y, X, W, P, R_all)

        # Rescale all variables before continuing
        rescale(W, P, R_all, G)

    if proj_back:
        z = projection_back(Y, X[:,:,0])
        Y *= np.conj(z[None,:,:])

    if return_filters:
        return Y, W, G, R_all
    else:
        return Y
