
import os
import itertools

import rrtools

# find the absolute path to this file
base_dir = os.path.abspath(os.path.split(__file__)[0])

def init(parameters):
    parameters['lower_bound'] = 0


def parallel_loop(args):
    global parameters

    import time

    # split arguments
    timeout = args[0]
    key = args[1]

    time.sleep(timeout)
    
    return dict(key=key, timeout=timeout, secret=parameters['secret'])

def gen_args(parameters):

    timeouts = range(parameters['max_timeout'])
    keys = range(parameters['max_int'])

    return list(itertools.product(timeouts, keys))


if __name__ == '__main__':

    rrtools.run(parallel_loop, gen_args, func_init=init, base_dir=base_dir, results_dir='data/', description='Dummy test simulation')

 

