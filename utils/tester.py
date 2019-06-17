""" Script for re-validating all checkpoints of a single model run on some specific validation set, followed by the
calculation of the test error using the single-best validation checkpoint. """

import os
import argparse
import subprocess
from collections import OrderedDict

from training_progress import TrainingProgress
from rescorer import get_checkpoint_prefixes


def test(ckpt_directory, script_path, progress_path, no_single_checkpoints, big_model, use_last, test_sets):
    """ Calculates the test error of a model on the specified test-sets. Validation script must be already set-up. """
    # Obtain the list of checkpoint prefixes for the specified model directory
    ckpt_prefixes = get_checkpoint_prefixes(ckpt_directory)
    model_name = ckpt_prefixes[0].split('-')[0]
    # Reload training progress
    progress = TrainingProgress()
    progress.load_from_json(progress_path)
    # Get the best-performing model checkpoints based on validation scores
    best_ppx_step = [int(min(progress.validation_perplexity, key=lambda key: progress.validation_perplexity[key]))]
    best_bleu_step = [int(max(progress.validation_bleu, key=lambda key: progress.validation_bleu[key]))]
    # Define the checkpoint window for averaging
    ppx_step_list = sorted([int(step) for step in list(progress.validation_perplexity.keys())])
    bleu_step_list = sorted([int(step) for step in list(progress.validation_bleu.keys())])

    best_ppx_index = ppx_step_list.index(best_ppx_step[0]) if not use_last else len(ppx_step_list) - 1
    best_bleu_index = bleu_step_list.index(best_bleu_step[0]) if not use_last else len(bleu_step_list) - 1

    ckpt_bound = 4 if not big_model else 15

    best_ppx_last_k = ppx_step_list[: best_ppx_index + 1] if best_ppx_index < ckpt_bound \
        else ppx_step_list[best_ppx_index - ckpt_bound: best_ppx_index + 1]

    best_bleu_last_k = bleu_step_list[: best_bleu_index + 1] if best_bleu_index < ckpt_bound \
        else bleu_step_list[best_bleu_index - ckpt_bound: best_bleu_index + 1]

    # Convert steps to checkpoint indices
    best_ppx_step = ['{:s}-{:d}'.format(model_name, step) for step in best_ppx_step]
    best_bleu_step = ['{:s}-{:d}'.format(model_name, step) for step in best_bleu_step]
    best_ppx_last_k = ['{:s}-{:d}'.format(model_name, step) for step in best_ppx_last_k]
    best_bleu_last_k = ['{:s}-{:d}'.format(model_name, step) for step in best_bleu_last_k]

    # Add progress entries for test metrics
    test_progress = TrainingProgress()
    test_progress.test_perplexity = OrderedDict()
    test_progress.test_perplexity_last_k = OrderedDict()
    test_progress.test_bleu = OrderedDict()
    test_progress.test_bleu_last_k = OrderedDict()

    scratch_path = os.path.join(ckpt_directory, 'scratch.txt')
    log_path = os.path.join(ckpt_directory, 'log.txt')

    all_checkpoints = [best_bleu_last_k] if no_single_checkpoints else [best_bleu_step, best_bleu_last_k]

    # Calculate test set error
    for test_set in test_sets:
        # for ckpts in [best_ppx_step, best_ppx_last_k, best_bleu_step, best_bleu_last_k]
        for ckpts in all_checkpoints:
            with open(script_path, 'r') as script:
                with open(scratch_path, 'w') as scratch:
                    for line in script:

                        if '--valid_source_dataset' in line:
                            # Replace the validation set designation in the validation script
                            line_prefix, line_suffix = line.split('--valid_source_dataset')
                            path_prefix, path_suffix = line_suffix.split('/')
                            name_prefix, name_suffix = \
                                path_suffix.split('.')[0], '.'.join(path_suffix.split('.')[1:])
                            new_name = '.'.join([test_set, name_suffix])
                            new_path = '/'.join([path_prefix, new_name])
                            new_line = '--valid_source_dataset'.join([line_prefix, new_path])
                            scratch.write(new_line)

                        elif '--valid_target_dataset' in line:
                            # Replace the validation set designation in the validation script
                            line_prefix, line_suffix = line.split('--valid_target_dataset')
                            path_prefix, path_suffix = line_suffix.split('/')
                            name_prefix, name_suffix = \
                                path_suffix.split('.')[0], '.'.join(path_suffix.split('.')[1:])
                            new_name = '.'.join([test_set, name_suffix])
                            new_path = '/'.join([path_prefix, new_name])
                            new_line = '--valid_target_dataset'.join([line_prefix, new_path])
                            scratch.write(new_line)

                        elif '--valid_gold_reference' in line:
                            # Replace the validation set designation in the validation script
                            line_prefix, line_suffix = line.split('--valid_gold_reference')
                            path_prefix, path_suffix = line_suffix.split('/')
                            name_prefix, name_suffix = \
                                path_suffix.split('.')[0], '.'.join(path_suffix.split('.')[1:])
                            new_name = '.'.join([test_set, name_suffix])
                            new_path = '/'.join([path_prefix, new_name])
                            new_line = '--valid_gold_reference'.join([line_prefix, new_path])
                            scratch.write(new_line)

                        elif '--reload' in line and not line.startswith('#'):
                            # Replace the checkpoint designation in the validation script
                            line_prefix, line_suffix = line.split('--reload')
                            path_prefix = line_suffix.split('/')[0]
                            new_path = ''
                            for ckpt_prefix in ckpts:
                                new_path += '/'.join([path_prefix, ckpt_prefix])
                            reload_line = '--reload'.join([line_prefix, new_path])
                            scratch.write(reload_line + '\n')

                        else:
                            scratch.write(line)

            with open(scratch_path, 'r') as scratch:
                with open(script_path, 'w') as script:
                    for line in scratch:
                        script.write(line)

            # Run the script and capture output
            print('-' * 10)
            print('Calculating test-set perplexity and BLEU using {:d}-best checkpoint(s) and test set {:s}'
                  .format(len(ckpts), test_set))

            subprocess.run([script_path, ckpt_directory.split('/')[-1]])

            log_lines = list()
            with open(log_path, 'r') as log:
                for line in log:
                    if len(line.strip()) > 0:
                        log_lines.append(line)

            vals = ':'.join(log_lines[-1].split(':')[1:]).strip()
            loss_sv, ppx_sv, bleu_sv = [subval.strip() for subval in vals.split('|')]
            ppx_val = float(ppx_sv.split(':')[-1].strip())
            bleu_val = float(bleu_sv.split(':')[-1].strip())

            # Update progress file
            if len(ckpts) == 1:
                test_progress.test_perplexity[test_set] = ppx_val
                test_progress.test_bleu[test_set] = bleu_val
            else:
                test_progress.test_perplexity_last_k[test_set] = ppx_val
                test_progress.test_bleu_last_k[test_set] = bleu_val
            # Save progress file
            test_progress.save_to_json(progress_path[: -5] + '.tested_sacreBLEU.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, help='path to the model checkpoint directory')
    parser.add_argument('--script_path', type=str, help='path to the set-up validation script')
    parser.add_argument('--progress_path', type=str, help='path to the training progress JSON file')
    parser.add_argument('--no_single_checkpoints', action='store_true', help='perform only checkpoint averaging')
    parser.add_argument('--big_model', action='store_true', help='averages 16 checkpoints, instead of 5')
    parser.add_argument('--use_last', action='store_true', help='for averaging, use last checkpoint instead of best')
    parser.add_argument('--test_sets', nargs='+', type=str, default=None, help='names of the test set (optional)')
    args = parser.parse_args()

    test(args.ckpt_dir, args.script_path, args.progress_path, args.no_single_checkpoints, args.big_model,
         args.use_last, args.test_sets)
