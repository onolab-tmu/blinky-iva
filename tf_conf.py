'''
This script test the time-frequency computation/latency
trade-off of doing convolution in the STFT domain
'''
import numpy as np
from scipy.signal import fftconvolve
import pyroomacoustics as pra

x = np.random.randn(128 * 100)
h = np.random.randn(128)

y_direct = fftconvolve(x, h)

X = pra.transform.analysis(x, 64, 64, zp_back=64)  # shape (nframes, nfreq)
H = pra.transform.analysis(h, 64, 64, zp_back=64)  # shape (nframes, nfreq)

Y = []
for f in range(X.shape[1]):
    Y.append(fftconvolve(X[:,f], H[:,f]))

y_tf = pra.transform.synthesis(np.array(Y).T, 64, 64, zp_back=64)

print('Maximum error:', np.max(np.abs(y_tf[:y_direct.shape[0]] - y_direct[:y_tf.shape[0]])))
