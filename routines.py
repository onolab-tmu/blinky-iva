import numpy as np
import pyroomacoustics as pra

class EncodingBlinkyArray(pra.MicrophoneArray):
    '''
    This is an abstraction for the sound to light sensors
    using frequency domain encoding

    Attributes
    ----------
    R: array_like (n_dim, n_sensors)
        The sensor locations
    fs: int
        The sampling frequency
    freq_lim: tuple length 2
        Limits of the band in which to filter the data
    downsampling: int, optional
        The length of averaging
    filters: array_like (n_sensors, filter_length)
        The encoding filters 
    '''

    def __init__(self, R, fs, freq_lim=None, downsampling=1, filters=None):
        pra.MicrophoneArray.__init__(self, R, fs)
        self.filters = filters
        self.downsampling = downsampling
        self.nfft = 2 * filters.shape[1]
        self.filters_fd = np.fft.rfft(filters, axis=-1, n=self.nfft).T / np.sqrt(self.nfft)

    def record(self, signals, fs):
        '''
        This simulates the recording of the signals by the microphones
        and the transformation from sound to light.
         
        Parameters
        ----------
        signals:
            An ndarray with as many lines as there are microphones.
        fs:
            the sampling frequency of the signals.
        '''

        self.raw_signals = signals

        X = pra.transform.analysis(
                signals.T,
                self.downsampling,
                self.downsampling,
                zp_back=self.nfft - self.downsampling,
                )
        X /= np.sqrt(self.nfft)

        # frequency domain filtering
        Y = X * self.filters_fd[None,]

        self.filtered_signals = pra.transform.synthesis(Y, self.downsampling, self.downsampling, zp_back=self.nfft - self.downsampling)

        # power averaging to get the blinky signals
        self.signals = np.mean(np.abs(Y) ** 2, axis=1).T

        self.power = np.mean(np.abs(X) ** 2, axis=1).T


def generate_filters(n_sensors, length, type='binary'):
    '''
    Generates random filters

    Parameters
    ----------
    n_sensors: int
        The number of sensors
    length: int
        The length of the filters
    type: str
        The type of filter to generate (binary or real)


    returns
    -------
    filters: array_like (n_sensors, lengths)
        The filters

    '''

    # the filters have to be symetric to make sure they have a real transfer function
    if length % 2 == 0:
        n_half = length // 2
    else:
        n_half = length // 2 + 1

    if type == 'binary':
        F_half = np.random.choice(2, size=(n_sensors, n_half))
    else:
        F_half = np.sqrt(np.random.rand(n_sensors, n_half))

    blocks = [F_half]
    if length % 2 == 0:
        blocks.append(np.zeros((n_sensors,1)))
    blocks.append(F_half[:,:0:-1])

    filters = np.concatenate(blocks, axis=1)

    return filters


def random_layout(vol_dim, n_mic, offset=None, seed=None):

    if seed is not None:
        rng_state = np.random.get_state()
        np.random.seed(seed)

    vol_dim = np.array(vol_dim)

    if offset is None:
        offset = np.zeros_like(vol_dim)

    vol_dim = vol_dim + np.array(offset)

    points = []
    for l,u in zip(offset, vol_dim):
        points.append(np.random.uniform(l, u, size=n_mic))

    if seed is not None:
        np.random.set_state(rng_state)

    return np.array(points)


def grid_layout(room_dim, n_mic, offset=None):
    '''
    Place the microphones equispaced on a grid
    '''

    area = np.prod(room_dim[:2])
    sq_L = np.sqrt(area / n_mic)

    mic_loc = []
    wb = int(np.floor(room_dim[0] / sq_L))
    hb = n_mic // wb

    n = 0
    x,y = 0, 0
    while n < n_mic:
        mic_loc.append([x, y, 0.])

        n = n + 1
        if n % wb == 0:
            x = 0
            y += sq_L
        else:
            x += sq_L
    mic_loc = np.array(mic_loc).T

    # center the microphones (x-y plan only)
    for i in range(2):
        mic_loc[i,:] += (room_dim[i] - mic_loc[i,:].max()) / 2

    if offset is not None:
        mic_loc += np.array([offset]).T

    return mic_loc


def semi_circle_layout(center, angle, distance, n, rot=None):
    '''
    Places n points on a semi circle covering an angle, at a distance from center
    '''

    center = np.array(center)

    angles = np.linspace(0, angle, n) + rot

    v = np.array([np.cos(angles), np.sin(angles),]) * distance

    if center.shape[0] == 3:
        v = np.concatenate([v, np.zeros((1,n))], axis=0)

    v += center[:,None]

    return v

