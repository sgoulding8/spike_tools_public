import os
import argparse
from pathlib import Path
import json

import scipy.io as sio
import numpy as np
import pandas as pd
import xarray as xr

from brainio_base.assemblies import NeuronRecordingAssembly
import brainio_collection

from spike_tools import IMAGES, filter_neuroids


def _reorder(psth, images, image_order, fixation_correct, max_rep_num):
    reordered_psth = []
    for _, img in enumerate(images):
        r = np.where(image_order == img)[0]
        r = r[np.where(fixation_correct[r] == 1)[0]]
        if len(r) > max_rep_num:
            r = r[:max_rep_num]
        reordered_psth.append(psth[r, :, :])
    reordered_psth = np.array(reordered_psth)
    # print(reordered_psth.shape)
    return reordered_psth


def fetch(experiment_name, monkey_name, date, output_dir):
    home_dir = Path('/braintree/data2/active/users/sachis/projects/') / experiment_name / 'monkeys' / monkey_name
    intan_dir = home_dir / 'intanproc'  # Neural data directory
    assert intan_dir.exists()
    mwk_dir = home_dir / 'mworksproc'  # Behavioral data directory
    assert mwk_dir.exists()

    session_dirs = [d for d in os.listdir(intan_dir) if d.startswith(f'{monkey_name}_{experiment_name}')]
    mwk_files = [f for f in os.listdir(mwk_dir) if f.startswith(f'{monkey_name}_{experiment_name}')]
    if date is not None:
        session_dirs = [d for d in session_dirs if date in d]
        mwk_files = [f for f in mwk_files if date in f]
    session_dirs.sort()
    mwk_files.sort()
    assert len(session_dirs) == len(mwk_files)
    assert [(intan_dir / d / 'psth').exists() for d in session_dirs]  # Check `psth` subfolder exists for each session

    all_psth, all_normalizer, all_baseline = [], [], []

    # Loop through all sessions
    for i, (mwk_file, session_dir) in enumerate(zip(mwk_files, session_dirs)):
        print(i, mwk_file, session_dir)

        # Load behavioral data
        mwk_data = sio.loadmat(mwk_dir / mwk_file, squeeze_me=True)
        fixation_correct = mwk_data['fixation_correct']
        image_order = mwk_data['image_order']

        # Get max repetition number (i.e. all experiment images have been shown for that many repetitions)
        correct_trials = image_order[np.where(fixation_correct == 1)]
        repetition_count = [len(correct_trials[correct_trials == img]) for img in
                            sum([IMAGES[experiment_name]['experiment_images'],
                             IMAGES[experiment_name]['normalizer_images'],
                             IMAGES[experiment_name]['baseline_images']], [])
                            ]
        max_rep_num = min(repetition_count)

        # Get all individual channel psth data
        files = [f for f in os.listdir(intan_dir / session_dir / 'psth') if f.startswith('amp')]
        files.sort()
        psth = np.array([sio.loadmat(os.path.join(intan_dir, session_dir, 'psth', f), squeeze_me=True, variable_names='psth')['psth'] for f in files])
        psth = np.moveaxis(psth, 0, -1)
        meta = [sio.loadmat(os.path.join(intan_dir, session_dir, 'psth', f), squeeze_me=True, variable_names='meta')['meta'] for f in files]

        # normalizer_idxs = np.where( np.isin(correct_trials, IMAGES[experiment_name]['normalizer_images']) )[0]
        # baseline_idxs = np.where(np.isin(correct_trials, IMAGES[experiment_name]['baseline_images']))[0]
        # experiment_idxs = np.where(np.isin(correct_trials, IMAGES[experiment_name]['experiment_images']))[0]

        reordered_psth_experiment = _reorder(psth, images=IMAGES[experiment_name]['experiment_images'],
                                             image_order=image_order, fixation_correct=fixation_correct,
                                             max_rep_num=max_rep_num)
        # reordered_psth_normalizer = _reorder(psth, images=IMAGES[experiment_name]['normalizer_images'],
        #                                      image_order=image_order, fixation_correct=fixation_correct,
        #                                      max_rep_num=max_rep_num)
        # reordered_psth_baseline = _reorder(psth, images=IMAGES[experiment_name]['baseline_images'],
        #                                      image_order=image_order, fixation_correct=fixation_correct,
        #                                      max_rep_num=max_rep_num)

        if i == 0:
            all_psth = reordered_psth_experiment
            # all_normalizer = reordered_psth_normalizer
            # all_baseline = reordered_psth_baseline
        else:
            all_psth = np.hstack((all_psth, reordered_psth_experiment))
            # all_normalizer = np.hstack((all_normalizer, reordered_psth_normalizer))
            # all_baseline = np.hstack((all_baseline, reordered_psth_baseline))

        print(all_psth.shape)

    # output_file = Path(output_dir) / f'{monkey_name}.rsvp.{experiment_name}.experiment_psth.raw.npy
    output_file = Path(output_dir) / f'{monkey_name}.rsvp.{experiment_name}.psth.npy'
    if not output_file.exists():
        np.save(output_file, all_psth)
    return


def make_xarray(experiment_name, monkey_name, data_dir, stimuli):
    data_dir = Path(data_dir)
    psth = np.load(data_dir / 'solo.rsvp.hvm.experiment_psth.raw.npy')  # Shaped images x repetitions x time_bins x channels

    # timebins = [[70, 170], [170, 270], [50, 100], [100, 150], [150, 200], [200, 250], [70, 270]]
    photodiode_delay = 30  # Delay recorded on photodiode is ~30ms
    timebase = np.arange(-100, 381, 10)  # PSTH from -100ms to 380ms relative to stimulus onset
    assert len(timebase) == psth.shape[2]
    # time_bin_coords = [str(_) + '-' + str(_ + 10) for _ in timebase]

    # Load image related meta data (id ordering differs from dicarlo.hvm)
    image_id = [x.split()[0][:-4] for x in open(data_dir.parent / 'image-metadata' / 'hvm_map.txt').readlines()]
    # Load neuroid related meta data
    neuroid_meta = pd.DataFrame(json.load(open(data_dir.parent / 'array-metadata' / 'mapping.json')))

    assembly = xr.DataArray(psth,
                            coords={'repetition': ('repetition', list(range(psth.shape[1]))),
                                    'time_bin_id': ('time_bin', list(range(psth.shape[2]))),
                                    'time_bin_start': ('time_bin', [(x - photodiode_delay) for x in timebase]),
                                    'time_bin_stop': ('time_bin', [(x + 10 - photodiode_delay) for x in timebase]),
                                    'image_id': ('image', image_id)},
                            dims=['image', 'repetition', 'time_bin', 'neuroid'])

    for column_name, column_data in neuroid_meta.iteritems():
        assembly = assembly.assign_coords(**{f'{column_name}': ('neuroid', list(column_data.values))})

    assembly = assembly.sortby(assembly.image_id)
    stimuli = stimuli.sort_values(by='image_id').reset_index(drop=True)
    for column_name, column_data in stimuli.iteritems():
        assembly = assembly.assign_coords(**{f'{column_name}': ('image', list(column_data.values))})
    assembly = assembly.sortby(assembly.id)  # Re-order by id to match dicarlo.hvm ordering

    # Collapse dimensions 'image' and 'repetitions' into a single 'presentation' dimension
    assembly = assembly.stack(presentation=('image', 'repetition')).reset_index('presentation')
    assembly = NeuronRecordingAssembly(assembly)

    # Filter noisy electrodes
    psth = np.load(data_dir / 'solo.rsvp.hvm.normalizer_psth.npy')
    t_cols = np.where((timebase >= (70 + photodiode_delay)) & (timebase < (170 + photodiode_delay)))[0]
    rate = np.mean(psth[:, :, t_cols, :], axis=2)
    normalizer_assembly = xr.DataArray(rate,
                                       coords={'repetition': ('repetition', list(range(rate.shape[1]))),
                                               'image_id': ('image', list(range(rate.shape[0]))),
                                               'id': ('image', list(range(rate.shape[0])))},
                                       dims=['image', 'repetition', 'neuroid'])
    for column_name, column_data in neuroid_meta.iteritems():
        normalizer_assembly = normalizer_assembly.assign_coords(
            **{f'{column_name}': ('neuroid', list(column_data.values))})
    normalizer_assembly = normalizer_assembly.stack(presentation=('image', 'repetition')).reset_index('presentation')
    normalizer_assembly = normalizer_assembly.drop('image')
    normalizer_assembly = normalizer_assembly.transpose('presentation', 'neuroid')
    normalizer_assembly = NeuronRecordingAssembly(normalizer_assembly)

    filtered_assembly = filter_neuroids(normalizer_assembly, 0.7)
    assembly = assembly.sel(neuroid=np.isin(assembly.neuroid_id, filtered_assembly.neuroid_id))
    assembly = assembly.transpose('presentation', 'neuroid', 'time_bin')

    # Add other experiment related info
    assembly.attrs['image_size_degree'] = 8
    assembly.attrs['stim_on_time_ms'] = 100

    if not (data_dir / f'{monkey_name}.rsvp.{experiment_name}.experiment_psth.raw.nc').exists():
        assembly.reset_index(['presentation', 'neuroid', 'time_bin'], inplace=True)
        assembly.to_netcdf(data_dir / f'{monkey_name}.rsvp.{experiment_name}.experiment_psth.raw.nc')

    return assembly


def plot_psth(experiment_name, monkey_name, data_dir):
    data_dir = Path(data_dir)
    data = NeuronRecordingAssembly(xr.open_dataarray(data_dir / f'{monkey_name}.rsvp.{experiment_name}.experiment_psth.nc'))

    from brainscore.benchmarks._neural_common import average_repetition
    data = average_repetition(data)
    print(data)
    #
    # import matplotlib.pyplot as plt
    # y = data.sel(neuroid_id='A-001', image_id='0015b49a190e9bce70b108b28dc1a0674d3c9e66').squeeze()
    # print(y)
    # x = np.arange(-100-30-30, 381-30-30, 10)
    # print(x)
    # plt.plot(x, y.data)
    # plt.show()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fetch raw PSTH file')
    parser.add_argument('--experiment', default='normalizers', type=str,
                        help='Experiment name')
    parser.add_argument('--monkey', default='oleo', type=str,
                        help='Monkey name')
    parser.add_argument('--date', default=None, type=str)
    parser.add_argument('--output_dir', default='/braintree/data2/active/users/sachis/database', type=str)

    args = parser.parse_args()

    fetch(experiment_name=args.experiment.lower(), monkey_name=args.monkey.lower(), date=args.date,
          output_dir=args.output_dir)
    # make_xarray(experiment_name=args.experiment.lower(), monkey_name=args.monkey.lower(),
    #             data_dir=args.output_dir, stimuli=brainio_collection.get_stimulus_set('dicarlo.hvm'))
    # plot_psth(experiment_name=args.experiment.lower(), monkey_name=args.monkey.lower(), data_dir=args.output_dir)
