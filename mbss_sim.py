'''
This file contains the code to run the more systematic simulation.
'''
import argparse, json, os
import numpy as np
import pyroomacoustics as pra

from routines import PlaySoundGUI, grid_layout, semi_circle_layout, random_layout, gm_layout
from blinkiva import blinkiva
from generate_samples import sampling, wav_read_center

# convergence monitoring callback
def convergence_callback(Y, n_targets, SDR, SIR, ref, framesize, win_s):
    from mir_eval.separation import bss_eval_sources
    y = pra.transform.synthesis(Y, framesize, framesize // 2, win=win_s,)

    if args.algo != 'blinkiva':
        new_ord = np.argsort(np.std(y, axis=0))[::-1]
        y = y[:,new_ord]

    m = np.minimum(y.shape[0]-framesize//2, ref.shape[1])
    sdr, sir, sar, perm = bss_eval_sources(
            ref[:n_sources_target,:m,0],
            y[framesize//2:m+framesize//2,:n_targets].T,
            )
    SDR.append(sdr)
    SIR.append(sir)

# algorithm parameters
def generate(n_targets, n_mics, rt60, sir, snr, wav_files, config):
    '''
    This function will prepare all the data to be processed

    The single argument is a dictionary containing all the parameters
    '''

    # get the simulation parameters from the json file
    # Simulation parameters
    n_repeat = config['n_repeat']
    seed = config['seed']
    fs = config['fs']

    n_interferers = config['n_interferers']
    n_blinkies = config['n_blinkies']
    ref_mic = config['ref_mic']
    room_dim = np.array(config['room_dim'])

    source_var = np.ones(n_targets)
    source_var[0] = config['weak_source_var']

    # total number of sources
    n_sources = n_interferers + n_targets

    # Fix RNG Seed!
    np.random.seed(seed)

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
            absorption=rt60['absorption'],
            max_order=rt60['max_order'],
            )

    # Place all the sound sources
    for sig, loc in zip(signals, source_locs.T,):
        room.add_source(loc, signal=sig)

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
    premix[:n_targets,:,:] *= np.sqrt(src_var[:,None,None])

    # compute noise variance
    sigma_n = np.sqrt(10 ** (- snr / 10) * np.sum(src_var))

    # now compute the power of interference signal needed to achieve desired SIR
    sigma_i = np.sqrt(10 ** (- sir / 10) * np.sum(src_var) / n_interferers)
    premix[n_tgt:,:,:] *= sigma_i

    # Mix down the recorded signals
    mix = np.sum(premix[:n_src,:], axis=0) + sigma_n * np.random.randn(*premix.shape[1:])

    return mix, premix


def one_loop(n_targets, n_mics, rt60, sir, snr, wav_files, config):

    # STFT parameters
    framesize = config['stft_params']['framesize']
    win_a = pra.hann(framesize)
    win_s = pra.transform.compute_synthesis_window(win_a, framesize // 2)

    # Generate the audio signals
    mix, premix = generate(n_targets, n_mics, rt60, sir, snr, wav_files, config)
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

    results = {}

    for name, kwargs in config['algorithm_kwargs'].items():

        results[name] = { 'SDR' : [], 'SIR' : [], }

        cb = lambda Y :  convergence_callback(
                Y, n_targets,
                results[name]['SDR'], results[name]['SIR'],
                ref, framesize, win_s,
                )

        if args.algo == 'auxiva':
            # Run AuxIVA
            Y = pra.bss.auxiva(
                    X_mics,
                    callback=cb,
                    **kwargs,
                    )

        elif args.algo == 'blinkiva':
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

    return results


def generate_arguments(config):
    ''' add stuff there now '''

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Runs the Multi-modal Blind Source Separation simulation.')
    subparsers = parser.add_subparsers(help='sub-command help')

    # create the parser for the "gen" command
    parser_gen = subparsers.add_parser('gen', help='Pre-generate the data for the simulation')
    parser_gen.add_argument('config', type=str, help='The JSON configuration file')
    parser_gen.set_defaults(func=generate)

    # magic!
    args = parser.parse_args()
    args.func(args)
