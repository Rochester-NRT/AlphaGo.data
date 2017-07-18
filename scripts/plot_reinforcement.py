#!/usr/bin/env python
from sys import argv
import json
from collections import defaultdict
import numpy as np
from operator import itemgetter
import matplotlib.pyplot as plt
from math import ceil

font = {'size': 16}
plt.rc('font', **font)


def boxcar_average(xs, ys, npts, window):
    out_xs = np.linspace(min(xs), max(xs), npts)
    out_ys = np.nan * np.ones(npts)
    for i, x in enumerate(out_xs):
        navg = 0
        yavg = 0
        for xx, yy in zip(xs, ys):
            if x - window / 2 <= xx and x + window / 2 >= xx:
                navg += 1
                yavg = yavg + (yy - yavg) / navg
        if navg > 0:
            out_ys[i] = yavg
        else:
            out_ys[i] = np.nan
    return out_xs, out_ys


if len(argv) > 1:
    with open(argv[1], 'r') as f:
        metadata = json.load(f)
        n_opponents = len(metadata['opponents'])

        if len(argv) > 2:
            prefix = argv[2]
        else:
            prefix = ''

        if len(argv) > 3:
            smooth_window = int(argv[3])
        else:
            smooth_window = 50

        # Plot win percentage over time
        series = defaultdict(lambda: [])  # map from opponent to list of (iteration, win_rate)

        for player, opponent_results in metadata["win_ratio"].iteritems():
            # player name is weights.xxxxx.hdf5
            itr = int(player[8:13])
            series[opponent_results[0]].append((itr, opponent_results[1]))

        for (opponent, s) in series.items():
            series[opponent] = sorted(s, key=itemgetter(0))

        plt.figure(figsize=[8, 5])
        colors = 'brgcmyk'
        for i, opponent in enumerate(sorted(series.keys())):
            results = series[opponent]
            times, ratios = zip(*results)
            plt.plot([min(times), max(times)], [.5, .5], '-', color='k')
            plt.plot(times, ratios, ',', color=colors[i % len(colors)])
            xx, yy = boxcar_average(times, ratios, len(metadata["win_ratio"]), smooth_window)
            plt.plot(xx, yy, '-', label=opponent, color=colors[i % len(colors)])
            plt.grid()
        plt.xlabel('iteration')
        plt.ylabel('win ratio')
        plt.legend(loc='best', fontsize=10, ncol=int(ceil(n_opponents / 6.)))
        plt.ylim(0, 1)
        plt.title('RL win ratio [LR = %.3e]' % metadata["learning_rate"])
        plt.savefig(prefix + 'ratio_over_time.png')
else:
    print "usage: python plot_reinforcement.py METADATA_FILE [SAVE_PREFIX]"
