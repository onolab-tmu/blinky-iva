import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt

from routines import generate_filters, EncodingBlinkyArray, grid_layout

# Simulation parameters
fs = 16000
absorption = 0.25
max_order = 17
n_blinkies = 23
framesize = 512

# Geometry of the room and location of sources and microphones
room_dim = np.array([10, 7.5, 3])
source_loc = np.array([2.51, 3.57, 1.7])

# Place the blinkies regularly in the room (2D plane)
mic_loc = grid_layout(room_dim, n_blinkies, 0.7)

# Create the room itself
room = pra.ShoeBox(room_dim, fs=fs, absorption=absorption, max_order=max_order)

# Place a source of white noise playing for 5 s
source_signal = np.random.randn(fs * 5)
room.add_source(source_loc, signal=source_signal)

# Place the microphone array
filters = generate_filters(n_blinkies, framesize, type='real')
room.add_microphone_array(
        EncodingBlinkyArray(mic_loc, fs=room.fs, downsampling=framesize, filters=filters)
        )

# Now the setup is finished, run the simulation
room.simulate()

'''
# Plot the room and the image sources
room.plot(img_order=0)
plt.title('The room with 4 generations of image sources')

# Plot the impulse responses
plt.figure()
room.plot_rir()

plt.show()
'''
