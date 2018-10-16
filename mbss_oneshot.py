import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt

from mir_eval.separation import bss_eval_sources
from routines import grid_layout
from blinkiva import blinkiva

# Simulation parameters
fs = 16000
absorption = 0.25
max_order = 17
n_blinkies = 23
framesize = 1024

# Geometry of the room and location of sources and microphones
room_dim = np.array([10, 7.5, 3])
source_loc = np.array([2.51, 3.57, 1.7])
mic_locs = np.c_[[2, 2.5, 1.6], [4, 6, 1.7]]
n_mics = mic_locs.shape[1]

# Place the blinkies regularly in the room (2D plane)
blinky_locs = grid_layout(room_dim, n_blinkies, 0.7)
all_locs = np.concatenate((mic_locs, blinky_locs), axis=1)

# Create the room itself
room = pra.ShoeBox(room_dim, fs=fs, absorption=absorption, max_order=max_order)

# Place a source of white noise playing for 5 s
source_signal = np.random.randn(fs * 5)
room.add_source(source_loc, signal=source_signal)

# Place the microphone array
room.add_microphone_array(
        pra.MicrophoneArray(all_locs, fs=room.fs)
        )

# Now the setup is finished, run the simulation
room.simulate()

# pad the output signal to a multiple of the framesize
signals = room.mic_array.signals
pad_shape = (signals.shape[0], framesize - signals.shape[1] % framesize)
signals = np.concatenate((signals, np.zeros(pad_shape)))

# STFT of microphone signals
win_a = pra.hann(framesize)
win_s = pra.transform.compute_synthesis_window(win_a, framesize // 2)
STFT stft(framesize, framesize // 2, analysis_window=win_a, synthesis_window=win_s, channels=n_mics+n_blinkies)

# shape: (n_frames, n_freq, n_mics)
X_all = pra.transform.analysis(
        signals.T,
        framesize, framesize // 2,
        win=win_a,
        ) 
X_mics =  X_all[:,:,:n_mics]
V_blinky = np.sum(np.abs(Y_all[:,:,n_mics:]) ** 2, axis=1)  # shape: (n_frames, n_blinkies)

Y, G, R = blinkiva(X_mics, V_blinky)



'''
# Plot the room and the image sources
room.plot(img_order=0)
plt.title('The room with 4 generations of image sources')

# Plot the impulse responses
plt.figure()
room.plot_rir()

plt.show()
'''
