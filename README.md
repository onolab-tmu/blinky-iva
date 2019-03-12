Multi-modal Blind Source Separation with Microphones and Blinkies
=================================================================

This repository provides implementations and code to reproduce the results
of the paper

> Robin Scheibler and Nobutaka Ono, *"Multi-modal Blind Source Separation with Microphones and Blinkies,"* Proc. IEEE ICASSP, Brighton UK, 2019.

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

Reproduce the Results
---------------------

The preferred way is to use [anaconda](https://www.anaconda.com/distribution/).

    conda env create -f environment.yml

The code can be run serially, or using multiple parallel workers via
[ipyparallel(https://ipyparallel.readthedocs.io/en/latest/).
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
