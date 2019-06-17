import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import json

def load_from_json(file_name):
    with open(file_name, 'r') as out_file:
        return json.load(out_file)


def plot_curves(json_logs, exp_ids, mode, factor):
    """ Plots perplexity/ BLEU curves of multiple experiments for comparison against each other """
    # Set up plot
    sns.set(style='whitegrid', context='paper', font_scale=1.3)
    sns.despine()
    plt.locator_params(axis='y', nbins=10)
    plt.locator_params(axis='x', nbins=10)
    # Parse logs
    exp_vals_all = list()
    time_steps_all = list()
    for _, log in enumerate(json_logs):
        bleu_and_ppx = load_from_json(log)
        val_dict = bleu_and_ppx['validation_{:s}'.format(mode)]
        # Extract keys from BLEU / perplexity dicts
        sorted_keys = [int(key) for key in val_dict.keys()]
        sorted_keys.sort()
        exp_vals = [val_dict[str(key)] for key in sorted_keys]
        time_steps = sorted_keys
        # Thin-out log data
        exp_vals_shrunk = list()
        time_steps_shrunk = list()
        step = 0
        if factor > 1:
            while (step + factor) < len(exp_vals):
                exp_vals_shrunk.append(sum(exp_vals[step: (step + factor)]) / factor)
                time_steps_shrunk.append(sum(time_steps[step: (step + factor)]) / factor)
                step += factor
            exp_vals = exp_vals_shrunk
            time_steps = time_steps_shrunk

        exp_vals_all.append(exp_vals)
        time_steps_all.append(time_steps)

    markers = list('+.x')    
    for step, bleu in enumerate(exp_vals_all):
        plt.plot(time_steps_all[step], bleu, label=exp_ids[step], marker=markers[step], markersize=8)

    loc = 'upper right' if mode == 'perplexity' else 'lower right'
    mode = mode if mode == 'perplexity' else mode.upper()

    plt.legend(loc=loc)
    # plt.title('{:s} per number of training updates'.format(mode), fontsize=13)
    plt.xlabel('num updates')
    plt.ylabel('validation {:s}'.format(mode))

    # Adjust plot margins (https://stackoverflow.com/questions/18619880/matplotlib-adjust-figure-margin)
    plot_margin = 0.25
    x0, x1, y0, y1 = plt.axis()
    plt.axis((x0 - plot_margin,
              x1 + plot_margin,
              y0 - plot_margin,
              y1 + plot_margin))

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logs', nargs='+', help='logs containing the plotted data', required=True)
    parser.add_argument('--exp_ids', nargs='+', help='experiment ids describing the plotted data', required=True)
    parser.add_argument('--mode', type=str, default='bleu', choices=['bleu', 'perplexity'], help='values to be plotted')
    parser.add_argument('--factor', type=int, default=2, help='reduction factor for the values array')
    args = parser.parse_args()

    plot_curves(args.logs, args.exp_ids, args.mode, args.factor)
