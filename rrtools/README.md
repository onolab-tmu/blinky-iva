Reproducible Research Tools
===========================

This is a helper tool to quickly build large dumb parallel simulation or
processing in a reproducible way.

* Parallelization is done using [ipyparallel](http://ipyparallel.readthedocs.io/en/latest/)
* Results are saved in human friendly JSON format, as soon as collected
* Provided the main project is versioned with git, the state of the repo
  is checked prior to simulation. If the repo is dirty, simulation is aborted
* The results are tagged with the commit number
* A basic interface displays how many loops have been done, how much time has ellapsed,
  approximately how long is left
* Options allow to run a single loop or in serial mode (without using ipyparallel)
  for debugging
* All the arguments and parameters for the simulation are saved along the results

Basics
------

The code to repeat is isolated in a function taking as single argument a list `args` .
This list `args` contains all the parameters that vary from loop to loop. The global
parameters, that are always the same during all loops of the simulation are stored in
a Python dictionary called `parameters`.

Every script created with `rrtools` comes with a list of options that can be
accessed through the help command

    $ python examples/test_simulation.py --help

    usage: test_simulation.py [-h] [-d DIR] [-p PROFILE] [-t] [-s] [--dummy]
                              parameters

    Dummy test simulation

    positional arguments:
      parameters            JSON file containing simulation parameters

    optional arguments:
      -h, --help            show this help message and exit
      -d DIR, --dir DIR     directory to store sim results
      -p PROFILE, --profile PROFILE
                            ipython profile of cluster
      -t, --test            test mode, runs a single loop of the simulation
      -s, --serial          run in a serial loop, ipyparallel not called
      --dummy               tags the directory as dummy, can be used for running
                            small batches

If using a cluster of `ipyparallel` engines is not available, it is possible
to run everything in a simple loop using the `-s` of `--serial` option.

For debugging, the `-t` or `--test` option runs only 2 loops of all.

Using the `--dummy` option will tag the results with `dummy` tag, which
is useful to make sure we distinguish test runs from the real simulation
results.

Example
-------

A simple example is availble in `examples` folder. It can be run like this

    python examles/test_simulation.py examples/test_simulation.json

The python file contains the function definitions for the different parts

    import os
    import itertools

    import rrtools

    # find the absolute path to this file
    base_dir = os.path.abspath(os.path.split(__file__)[0])

    def init(parameters):
        '''
        This function takes as unique positional argument a Python
        dictionary of global parameters for the simulation.
        This lets the user add some parameters computed in software
        to the dictionary. The update dictionary will be saved
        along the simulation output.

        This updated dictionary is later availbable in the global namespace of
        parallel_loop and gen_args functions.

        Parameters
        ----------
        parameters: dict
          The global simulation parameters
        '''
        parameters['lower_bound'] = 0


    def parallel_loop(args):
        '''
        This is the heart of the parallel simulation. This function is what is repeated
        a large number of time.

        Parameters
        ----------
        args: list
            A list of arguments whose combination is unique to one loop of the simulation.
        '''
        global parameters
        import time

        # split arguments
        timeout = args[0]
        key = args[1]

        time.sleep(timeout)
        
        return dict(key=key, timeout=timeout, secret=parameters['secret'])

    def gen_args(parameters):
        '''
        This function is called once before the simulation to generate
        the list of arguments combinations to try.

        For example say that you have arguments x=1,2,3 and y=2,3 for your parallel
        loop and you want to try all combinations. Then this function
        can generate the list
        args = [[1,2], [1,3], [2,2], [2,3], [3,2], [3,3]]

        Paramters
        ---------
        parameters: dict
            The Python dictionary of globaly simulation parameters. This can
            typically contain the range of values for the arguments to sweep.
        '''

        timeouts = range(parameters['max_timeout'])
        keys = range(parameters['max_int'])

        return list(itertools.product(timeouts, keys))


    if __name__ == '__main__':

        rrtools.run(parallel_loop, gen_args, func_init=init,
            base_dir=base_dir, results_dir='data/',
            description='Dummy test simulation')


The JSON file contains global simulation parameters.

    {
      "max_timeout": 10,
      "max_int": 2,
      "secret": "helloworld"
    }


Author
------

Robin Scheibler [contact](mailto://fakufaku@gmail.com)

License
-------

Copyright (c) 2018 Robin Scheibler

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
