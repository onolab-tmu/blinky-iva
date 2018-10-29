
import sys, argparse, os, json
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':

    # parse arguments
    parser = argparse.ArgumentParser(
                        description='Plot the data simulated by separake_near_wall')
    parser.add_argument('-p', '--pickle', action='store_true',
                        help='Read the aggregated data table from a pickle cache')
    parser.add_argument('-s', '--show', action='store_true',
                        help='Display the plots at the end of data analysis')
    parser.add_argument('dirs', type=str, nargs='+', metavar='DIR',
                        help='The directory containing the simulation output files.')

    cli_args = parser.parse_args()
    plot_flag = cli_args.show
    pickle_flag = cli_args.pickle

    parameters = dict()
    algorithms = dict()
    args = []
    df = None

    data_files = []

    for i, data_dir in enumerate(cli_args.dirs):

        print('Reading in', data_dir)

        # add the data file from this directory
        data_file = os.path.join(data_dir, 'data.json')
        if os.path.exists(data_file):
            data_files.append(data_file)
        else:
            raise ValueError('File {} doesn''t exist'.format(data_file))

        # get the simulation config
        with open(os.path.join(data_dir, 'parameters.json'), 'r') as f:
                parameters = json.load(f)


    # algorithms to take in the plot
    algos = algorithms.keys()

    # check if a pickle file exists for these files
    pickle_file = '.mbss.pickle'

    if os.path.isfile(pickle_file) and pickle_flag:
        print('Reading existing pickle file...')
        # read the pickle file
        perf = pd.read_pickle(pickle_file)

    else:

        # reading all data files in the directory
        records = []
        for file in data_files:
            with open(file, 'r') as f:
                content = json.load(f)
                for seg in content:
                    records += seg

        # build the data table line by line
        print('Building table')
        columns = [
                'Algorithm', 'Sources', 'Mics',
                'RT60', 'SINR', 'seed',
                'Strength', 'SDR', 'SIR',
                ]
        table = []
        num_sources = set()

        copy_fields = ['algorithm', 'n_targets', 'n_mics', 'rt60', 'sinr', 'seed']

        for record in records:

            entry = [ record[field] for field in copy_fields ]

            table.append(entry + [ 'Weak', record['sdr'][-1][0], record['sir'][-1][0] ])
            table.append(entry + [
                'Strong (avg.)', np.mean(record['sdr'][-1][1:]), np.mean(record['sir'][-1][1:]),
                ])
            table.append(entry + [
                'All sources (avg.)', np.mean(record['sdr'][-1]), np.mean(record['sir'][-1]),
                ])

        # create a pandas frame
        print('Making PANDAS frame...')
        df = pd.DataFrame(table, columns=columns)

        df.to_pickle(pickle_file)

    # Draw the figure
    print('Plotting...')

    # sns.set(style='whitegrid')
    # sns.plotting_context(context='poster', font_scale=2.)
    # pal = sns.cubehelix_palette(8, start=0.5, rot=-.75)

    df = df.replace(
            {
                'Algorithm' : {
                    'blinkiva-gauss' : 'BlinkIVA',
                    'auxiva' : 'AuxIVA',
                    }
                },
            )

    sns.set(style='whitegrid', context='paper', font_scale=1.,
            rc={
                #'figure.figsize': (3.5, 3.15),
                'lines.linewidth': 1.,
                #'font.family': 'sans-serif',
                #'font.sans-serif': [u'Helvetica'],
                'text.usetex': False,
            })
    pal = sns.cubehelix_palette(4, start=0.5, rot=-0.5, dark=0.3,
                                light=.75, reverse=True, hue=1.)
    sns.set_palette(pal)

    fig_dir = 'figures/{}_{}_{}'.format(
            parameters['name'], parameters['_date'], parameters['_git_sha'],
            )

    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)

    fn_tmp = os.path.join(fig_dir, 'RT60_{rt60}_SINR_{sinr}_{metric}.pdf')

    for rt60 in parameters['rt60_list']:
        for sinr in parameters['sinr_list']:

            select = np.logical_and(
                    df['RT60'] == rt60,
                    df['SINR'] == sinr,
                    )

            for metric in ['SDR', 'SIR']:

                g = sns.factorplot(data=df[select],
                        x='Mics', y=metric,
                        hue='Algorithm', col='Strength', row='Sources',
                        col_order=['All sources (avg.)', 'Weak', 'Strong (avg.)',],
                        hue_order=['AuxIVA','BlinkIVA'], kind='box',
                        legend=False,
                        #size=3, aspect=0.65,
                        )


                left_ax = g.facet_axis(0,0)
                leg = left_ax.legend(
                        title='Algorithm', frameon=True,
                        framealpha=0.85, fontsize='xx-small', loc='upper left',
                        )
                leg.get_frame().set_linewidth(0.0)

                sns.despine(offset=10, trim=False, left=True, bottom=True)

                plt.tight_layout(pad=0.01)

                plt.subplots_adjust(top=0.9)
                tit = g.fig.suptitle('# blinkies={}, RT60={}, SINR={}'.format(
                    parameters['n_blinkies'], rt60, sinr
                    ))

                fig_fn = fn_tmp.format(rt60=rt60, sinr=sinr, metric=metric)
                plt.savefig(fig_fn, bbox_extra_artists=(tit, leg,), bbox_inches='tight')

    if plot_flag:
        plt.show()
