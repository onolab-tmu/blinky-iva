'''
This file contains the code to run the more systematic simulation.
'''
import argparse, json, os
import numpy as np
import pyroomacoustics as pra
import rrtools

from routines import PlaySoundGUI, grid_layout, semi_circle_layout, random_layout, gm_layout
from blinkiva import blinkiva
from generate_samples import sampling, wav_read_center

# find the absolute path to this file
base_dir = os.path.abspath(os.path.split(__file__)[0])

# algorithm parameters
def generate(n_targets, n_mics, rt60, sinr, wav_files, config):
    '''
    This function will prepare all the data to be processed

    The single argument is a dictionary containing all the parameters
    '''

    return mix, premix


def one_loop(args):

    n_targets, n_mics, rt60, sinr, wav_files, seed = args

    # set MKL to only use one thread if present
    try:
        import mkl
        mkl.set_num_threads(1)
    except ImportError:
        pass

    # set the RNG seed
    rng_state = np.random.get_state()
    np.random.seed(seed)

    # STFT parameters
    framesize = parameters['stft_params']['framesize']
    win_a = pra.hann(framesize)
    win_s = pra.transform.compute_synthesis_window(win_a, framesize // 2)

    # Generate the audio signals

    # get the simulation parameters from the json file
    # Simulation parameters
    n_repeat = parameters['n_repeat']
    fs = parameters['fs']
    snr = parameters['snr']

    n_interferers = parameters['n_interferers']
    n_blinkies = parameters['n_blinkies']
    ref_mic = parameters['ref_mic']
    room_dim = np.array(parameters['room_dim'])

    sources_var = np.ones(n_targets)
    sources_var[0] = parameters['weak_source_var']

    # total number of sources
    n_sources = n_interferers + n_targets

    # Geometry of the room and location of sources and microphones
    interferer_locs = random_layout([3., 5.5, 1.5], n_interferers, offset=[6.5, 1., 0.5], seed=1)

    blinky_locs = semi_circle_layout(
            [4.1, 3.755, 1.1],
            np.pi, 3.5,
            n_blinkies,
            rot=0.743 * np.pi - np.pi / 4,
            )
    blinky_locs += np.random.randn(*blinky_locs.shape) * 0.01  # few millimeters shift

    mic_locs = np.vstack((
        pra.circular_2D_array([4.1, 3.76], n_mics, np.pi / 2, 0.02),
        1.2 * np.ones((1, n_mics)),
        ))
    all_locs = np.concatenate((mic_locs, blinky_locs), axis=1)

    target_locs = semi_circle_layout(
            [4.1, 3.755, 1.1],
            np.pi / 2, 2.,
            n_targets,
            rot=0.743 * np.pi,
            )
    source_locs = np.concatenate((target_locs, interferer_locs), axis=1)

    signals = wav_read_center(wav_files, seed=123)

    # Create the room itself
    room = pra.ShoeBox(room_dim,
            fs=fs,
            absorption=parameters['rt60_list'][rt60]['absorption'],
            max_order=parameters['rt60_list'][rt60]['max_order'],
            )

    # Place all the sound sources
    for sig, loc in zip(signals[-n_sources:,:], source_locs.T,):
        room.add_source(loc, signal=sig)

    assert signals.shape[0] == n_sources, 'Number of signals doesn''t match number of sources'

    # Place the microphone array
    room.add_microphone_array(
            pra.MicrophoneArray(all_locs, fs=room.fs)
            )

    # compute RIRs
    room.compute_rir()

    # Run the simulation
    premix = room.simulate(return_premix=True)

    # Normalize the signals so that they all have unit
    # variance at the reference microphone
    p_mic_ref = np.std(premix[:,ref_mic,:], axis=1)
    premix /= p_mic_ref[:,None,None]

    # scale to pre-defined variance
    premix[:n_targets,:,:] *= np.sqrt(sources_var[:,None,None])

    # compute noise variance
    sigma_n = np.sqrt(10 ** (- snr / 10) * np.sum(sources_var))

    # now compute the power of interference signal needed to achieve desired SINR
    sigma_i = np.sqrt(
            np.maximum(0, 10 ** (- sinr / 10) * np.sum(sources_var) - sigma_n ** 2) / n_interferers
            )
    premix[n_targets:,:,:] *= sigma_i

    # Mix down the recorded signals
    mix = np.sum(premix, axis=0) + sigma_n * np.random.randn(*premix.shape[1:])

    ref = np.moveaxis(premix, 1, 2)
    
    # START BSS
    ###########

    # shape: (n_frames, n_freq, n_mics)
    X_all = pra.transform.analysis(
            mix.T,
            framesize, framesize // 2,
            win=win_a,
            ) 
    X_mics =  X_all[:,:,:n_mics]
    U_blinky = np.sum(np.abs(X_all[:,:,n_mics:]) ** 2, axis=1)  # shape: (n_frames, n_blinkies)

    # convergence monitoring callback
    def convergence_callback(Y, n_targets, SDR, SIR, ref, framesize, win_s, algo_name):
        from mir_eval.separation import bss_eval_sources
        y = pra.transform.synthesis(Y, framesize, framesize // 2, win=win_s,)

        if algo_name != 'blinkiva':
            new_ord = np.argsort(np.std(y, axis=0))[::-1]
            y = y[:,new_ord]

        m = np.minimum(y.shape[0]-framesize//2, ref.shape[1])
        sdr, sir, sar, perm = bss_eval_sources(
                ref[:n_targets,:m,0],
                y[framesize//2:m+framesize//2,:n_targets].T,
                )
        SDR.append(sdr.tolist())
        SIR.append(sir.tolist())

    # store results in a list, one entry per algorithm
    results = []

    for name, kwargs in parameters['algorithm_kwargs'].items():

        results.append({ 
                'algorithm' : name,
                'n_targets' : n_targets,
                'n_mics' : n_mics,
                'rt60' : rt60,
                'sinr' : sinr,
                'seed' : seed,
                'sdr' : [], 'sir' : [],  # to store the result
                })

        cb = lambda Y :  convergence_callback(
                Y, n_targets,
                results[-1]['sdr'], results[-1]['sir'],
                ref, framesize, win_s, name,
                )

        if name == 'auxiva':
            # Run AuxIVA
            Y = pra.bss.auxiva(
                    X_mics,
                    callback=cb,
                    **kwargs,
                    )

        elif name == 'blinkiva':
            # Run BlinkIVA
            Y = blinkiva(
                    X_mics, U_blinky, n_src=n_targets,
                    callback=cb,
                    **kwargs,
                    )

        else:
            continue

        # The last evaluation
        cb(Y)

    # restore RNG former state
    np.random.set_state(rng_state)

    return results


def generate_arguments(parameters):
    ''' This will generate the list of arguments to run simulation for '''

    rng_state = np.random.get_state()
    np.random.seed(parameters['seed'])

    gen_files_seed = int(np.random.randint(2 ** 32, dtype=np.uint32))
    all_wav_files = sampling(
            parameters['n_repeat'],
            parameters['n_interferers'] + np.max(parameters['n_targets_list']),
            parameters['samples_list'],
            gender_balanced=True,
            seed=gen_files_seed,
            )

    args = []

    for n_targets in parameters['n_targets_list']:
        for n_mics in parameters['n_mics_list']:
            for rt60 in parameters['rt60_list'].keys():
                for sinr in parameters['sinr_list']:
                    for wav_files in all_wav_files:

                        # generate the seed for this simulation
                        seed = int(np.random.randint(2 ** 32, dtype=np.uint32))

                        # add the new combination to the list
                        args.append([n_targets, n_mics, rt60, sinr, wav_files, seed])
    
    np.random.set_state(rng_state)

    return args


if __name__ == '__main__':

    rrtools.run(
            one_loop,
            generate_arguments,
            base_dir=base_dir,
            results_dir='data/',
            description='Simulation for Multi-modal BSS with blinkies (ICASSP 2019)',
            )
