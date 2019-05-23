Multi-modal Blind Source Separation with Microphones and Blinkies
=================================================================

This repository provides implementations and code to reproduce the results
of the paper

> Robin Scheibler and Nobutaka Ono, [*"Multi-modal Blind Source Separation with Microphones and Blinkies,"*](https://arxiv.org/abs/1904.02334) Proc. IEEE ICASSP, Brighton UK, 2019.

Abstract
--------

We propose a blind source separation algorithm that jointly exploits
measurements by a conventional microphone array and an ad hoc array of low-rate
sound power sensors called blinkies. While providing less information than
microphones, blinkies circumvent some difficulties of microphone arrays in
terms of manufacturing, synchronization, and deployment. The algorithm is
derived from a joint probabilistic model of the microphone and sound power
measurements. We assume the separated sources to follow a time-varying
spherical Gaussian distribution, and the non-negative power measurement
space-time matrix to have a low-rank structure. We show that alternating
updates similar to those of independent vector analysis and Itakura-Saito
non-negative matrix factorization decrease the negative log-likelihood of the
joint distribution. The proposed algorithm is validated via numerical
experiments. Its median separation performance is found to be up to 8 dB more
than that of independent vector analysis, with significantly reduced
variability.

Authors
-------

[Robin Scheibler](http://robinscheibler.org) and [Nobutaka
Ono](http://www.comp.sd.tmu.ac.jp/onolab/index-e.html) are with the Faculty of
System Design at [Tokyo Metropolitan University](https://www.tmu.ac.jp/english/index.html).

### Contact

    Robin Scheibler (robin[at]tmu[dot]ac[dot]jp)
    6-6 Asahigaoka
    Hino, Tokyo
    191-0065 Japan

Preliminaries
-------------

The preferred way to run the code is using [anaconda](https://www.anaconda.com/distribution/).
An `environment.yml` file is provided to install the required dependencies.

    # create the minimal environment
    conda env create -f environment.yml

    # switch to new environment
    conda activate 2019_scheibler_icassp_blinkiva

Test BlinkIVA
-------------

The algorithm can be tested and compared to others using the sample
script `mbss_oneshot.py`. It can be run as follows.

    $ python ./mbss_oneshot.py --help
    usage: mbss_oneshot.py [-h] [-b BLOCK] [-m MICS] [-s {1,2,3,4,5,6,7,8,9,10}]
                           [-a {blinkiva,ilrma,auxiva,auxiva-gauss}] [--gui]
                           [--save]

    Demonstration of blind source separation using microphones and blinkies.

    optional arguments:
      -h, --help            show this help message and exit
      -b BLOCK, --block BLOCK
                            STFT block size
      -m MICS, --mics MICS  Number of microphones
      -s {1,2,3,4,5,6,7,8,9,10}, --srcs {1,2,3,4,5,6,7,8,9,10}
                            Number of sources
      -a {blinkiva,ilrma,auxiva,auxiva-gauss}, --algo {blinkiva,ilrma,auxiva,auxiva-gauss}
                            Chooses BSS method to run
      --gui                 Creates a small GUI for easy playback of the sound
                            samples
      --save                Saves the output of the separation to wav files

For example, we can run BlinkIVA with 4 microphones and 2 sources.

    python ./mbss_oneshot.py -a blinkiva -m 4 -s 2

Reproduce the Results
---------------------

The code can be run serially, or using multiple parallel workers via
[ipyparallel](https://ipyparallel.readthedocs.io/en/latest/).
Moreover, it is possible to only run a few loops to test whether the
code is running or not.

1. Run **test** loops **serially**

        python ./mbss_sim.py ./mbss_sim_config.json -t -s

2. Run **test** loops in **parallel**

        # start workers in the background
        # N is the number of parallel process, often "# threads - 1"
        ipcluster start --daemonize -n N

        # run the simulation
        python ./mbss_sim.py ./mbss_sim_config.json -t

        # stop the workers
        ipcluster stop

3. Run the whole simulation

        # start workers in the background
        # N is the number of parallel process, often "# threads - 1"
        ipcluster start --daemonize -n N

        # run the simulation
        python ./mbss_sim.py ./mbss_sim_config.json

        # stop the workers
        ipcluster stop

The results are saved in a new folder `data/<data>-<time>_mbss_sim_<flag_or_hash>`
containing the following files

    parameters.json  # the list of global parameters of the simulation
    arguments.json  # the list of all combinations of arguments simulated
    data.json  # the results of the simulation

Figure 3. from the paper is produced then by running

    python ./mbss_sim_plot.py data/<data>-<time>_mbss_sim_<flag_or_hash> -s

Data
----

For the experiment, we concatenated utterances from the CMU ARCTIC speech corpus to
obtain samples of at least 15 seconds long. The dataset thus created was stored on zenodo
with DOI [10.5281/zenodo.3066488](https://zenodo.org/record/3066489). The data is automatically
retrieved upon running the scripts, but can also be manually downloaded with the `get_data.py` script.

    python ./get_data.py

It is stored in the `samples` directory.

Use BlinkIVA
------------

Our implementation of BlinkIVA can be found in the file `blinkiva_gauss.py`.
It can be used as follows.

    from blinkiva_gauss import blinkiva_gauss

    # STFT tensor, a numpy.ndarray with shape (frames, frequencies, channels)
    X = ...

    # The blinky matrix is an numpy.ndarray with shape (frames, blinkies)
    U = ...

    # perform separation, output Y has the same shape as X
    Y = blinkiva_gauss(X, U, n_src=2, n_iter=50)

The function comes with docstrings.

    blinkiva_gauss(X, U, n_src=None, n_iter=20, n_nmf_sub_iter=20, proj_back=True,
                   W0=None, R0=None, seed=None, epsilon=0.5, sparse_reg=0.,
                   print_cost=False, return_filters=False, callback=None)

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
    n_nmf_sub_iter: int, optional
        The number of NMF iteration to run between two updates of the demixing
        matrices (default 20)
    proj_back: bool, optional
        Scaling on first mic by back projection (default True)
    W0: ndarray (nfrequencies, nchannels, nchannels), optional
        Initial value for demixing matrix
    R0: ndarray (nframes, nsrc), optional
        Initial value of the activations
    seed: int, optional
        A seed to make deterministic the random initialization of NMF parts,
        when None (default), the random number generator is used in its current state
    epsilon: float, optional
        A regularization value to prevent too large values after the division
    sparse_reg: float
        A regularization term to make the activation matrix sparse
    print_cost: bool, optional
        Print the value of the cost function at each iteration
    return_filters: bool, optional
        If true, the function will return the demixing matrix, gains, and activations too
    callback: func
        A callback function called every 10 iterations, allows to monitor convergence

    Returns
    -------
    Returns an (nframes, nfrequencies, nsources) array. Also returns
    the demixing matrix (nfrequencies, nchannels, nsources), gains (nsrc, nblinkies),
    and activations (nframes, nchannels) if ``return_filters`` keyword is True.

Summary of the Files in this Repo
---------------------------------

    environment.yml  # anaconda environment file

    auxiva_gauss.py  # implementation of auxiva with time-varying Gauss source model
    blinkiva_gauss.py  # implementation of blind source separation with microphones and blinkies
    get_data.py  # script that gets the data necessary for the experiment
    routines.py  # contains a bunch of helper routines for the simulation

    mbss_oneshot.py  # test file for source separation, with audible output
    mbss_sim.py  # script to run exhaustive simulation, used for the paper
    mbss_sim_config.json  # simulation configuration file
    mbss_sim_plot.py  # plots the figures from the output of overiva_sim.py

    data  # directory containing simulation results
    rrtools  # tools for parallel simulation
