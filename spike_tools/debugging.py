import os
from pathlib import Path
import re
import datetime

import numpy as np
import pandas as pd
import scipy.io as sio
import xarray as xr

import matplotlib.pyplot as plt
import seaborn as sns

from brainio_base.assemblies import NeuronRecordingAssembly

from spike_tools import IMAGES, filter_neuroids


def _annotate(data, **kws):
    n = data['Impedance Magnitude at 1000 Hz (ohms)'].values[0]
    ax = plt.gca()
    ax.text(.1, .6, f'{n/1000.} kohms', transform=ax.transAxes, fontsize=7)


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


def plot_impedance(monkey_name, surgery_date, data_dir=Path('/braintree/data2/active/users/sachis/impedances/monkeys')):
    assert data_dir.exists(), 'Invalid data directory'
    assert (data_dir / monkey_name).exists(), f'Invalid monkey name {monkey_name}'
    files = [f for f in os.listdir(data_dir / monkey_name) if not f.startswith('.')]
    files = sorted(files, key=lambda x: x.split('_')[-1].split('.')[0])  # Sort by date

    data = []
    for f in files:
        assert f.endswith(('.csv')), f'File {f} not of type CSV'
        assert re.match(f"{monkey_name}_" + "impedance_(\d{4}0?([1-9]|1[012])(0?[1-9]|[12][0-9]|3[01])).csv", f),\
            f'Invalid file name {f}'
        date = f.split('_')[-1].split('.')[0]
        date = datetime.date(int(date[0:4]), int(date[4:6]), int(date[6:]))
        # time.append((date - surgery_date).days)  # x-axis

        df = pd.read_csv(data_dir / monkey_name / f)
        df['Days Since Surgery'] = (date - surgery_date).days
        data.append(df)

    pd.set_option('display.max_columns', None)
    data = pd.concat(data, ignore_index=True)
    print(data)

    g = sns.FacetGrid(data, col='Channel Number', height=2, col_wrap=32, hue='Port', palette='tab20c')
    g.map(plt.plot, 'Days Since Surgery', 'Impedance Magnitude at 1000 Hz (ohms)', marker='o').set(yscale='log')

    # Draw a horizontal line to show the starting and end points for ideal impedances (100-800kohm)
    g.map(plt.axhline, y=100000, ls=':', c='.5')
    g.map(plt.axhline, y=800000, ls=':', c='.5')

    # Adjust the arrangement of the plots
    g.fig.tight_layout(w_pad=1, h_pad=2)

    plt.show()
    return


def plot_voltage(experiment_name, monkey_name, session_date):
    return


def plot_psth(experiment_name, monkey_name, session_date, show_impedance=True):
    intan_data_dir = Path(f'/braintree/data2/active/users/sachis/projects/{experiment_name}/monkeys/{monkey_name}/intanproc')
    mwk_data_dir = Path(f'/braintree/data2/active/users/sachis/projects/{experiment_name}/monkeys/{monkey_name}/mworksproc')
    assert intan_data_dir.exists()
    assert mwk_data_dir.exists()

    session_dirs = [intan_data_dir / f / 'psth' for f in os.listdir(intan_data_dir) if session_date in f]
    session_dirs.sort()
    mwk_files = [f for f in os.listdir(mwk_data_dir) if session_date in f]
    mwk_files.sort()

    assert len(session_dirs) == len(mwk_files)
    if len(mwk_files) == 0:
        return

    # Get impedance values for that date
    if show_impedance:
        impedance_file = Path(f'/braintree/data2/active/users/sachis/impedances/monkeys/{monkey_name}/{monkey_name}_impedance_20{session_date}.csv')
        assert impedance_file.exists(), f'Invalid file {impedance_file}'
        impedance_df = pd.read_csv(impedance_file)

    psth = []
    for i, (mwk_file, session_dir) in enumerate(zip(mwk_files, session_dirs)):
        # Load behavioral data
        mwk_data = sio.loadmat(mwk_data_dir / mwk_file, squeeze_me=True)
        fixation_correct = mwk_data['fixation_correct']
        image_order = mwk_data['image_order']

        # Get max repetition number (i.e. all experiment images have been shown for that many repetitions)
        correct_trials = image_order[np.where(fixation_correct == 1)]
        repetition_count = [len(correct_trials[correct_trials == img]) for img in range(1, 27)]  # TODO: remove hard-coding -- only works for normalizers
        max_rep_num = min(repetition_count)
        print(f'{max_rep_num} repeats')

        # Get all individual channel psth data
        intan_files = [f for f in os.listdir(session_dir) if f.startswith('amp')]
        intan_files.sort()
        assert len(intan_files) == 288  # TODO: Remove hard-coding
        psth_data = np.array(
            [sio.loadmat(session_dir / f, squeeze_me=True, variable_names='psth')['psth'] for f in intan_files])
        psth_data = np.moveaxis(psth_data, 0, -1)
        meta = [sio.loadmat(session_dir / f, squeeze_me=True, variable_names='meta')['meta'] for f in intan_files]

        reordered_psth = _reorder(psth_data,
                                  images=list(range(1, 27)),  # TODO: Remove hard-coding
                                  image_order=image_order, fixation_correct=fixation_correct,
                                  max_rep_num=max_rep_num)

        if i == 0:
            psth = reordered_psth
        else:
            psth = np.hstack((psth, reordered_psth))
    print(f'PSTH {psth.shape}')

    time = np.arange(-100, 390, 10)
    data = []
    for ch in range(psth.shape[-1]):
        curr_psth = np.mean(psth, axis=1)
        for image_id in range(psth.shape[0]):
            _curr_psth = curr_psth[image_id, :, ch]
            df = pd.DataFrame({
                'Time relative to SO (ms)': time,
                'Firing Rate': _curr_psth.tolist(),
            })
            df['Image'] = image_id
            df['Channel Number'] = ch
            if show_impedance:
                df['Channel Number'] = impedance_df.iloc[ch]['Channel Number']
                df['Impedance Magnitude at 1000 Hz (ohms)'] = impedance_df.iloc[ch]['Impedance Magnitude at 1000 Hz (ohms)']

            data.append(df)
    data = pd.concat(data, ignore_index=True)

    g = sns.FacetGrid(data, col='Channel Number', height=2, col_wrap=32, hue='Image', palette='gray')
    g.map(plt.plot, 'Time relative to SO (ms)', 'Firing Rate', lw=0.3)
    g.map_dataframe(_annotate)

    # Adjust the arrangement of the plots
    g.fig.tight_layout(w_pad=1, h_pad=2)

    plt.show()
    return


def fetch_normalizers(monkey_name, output_dir=None, date=None):
    experiment_name = 'normalizers'
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
                            sum([IMAGES[experiment_name]['normalizer_images'],
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

        # reordered_psth_experiment = _reorder(psth, images=IMAGES[experiment_name]['experiment_images'],
        #                                      image_order=image_order, fixation_correct=fixation_correct,
        #                                      max_rep_num=max_rep_num)
        reordered_psth_normalizer = _reorder(psth, images=IMAGES[experiment_name]['normalizer_images'],
                                             image_order=image_order, fixation_correct=fixation_correct,
                                             max_rep_num=max_rep_num)
        reordered_psth_baseline = _reorder(psth, images=IMAGES[experiment_name]['baseline_images'],
                                             image_order=image_order, fixation_correct=fixation_correct,
                                             max_rep_num=max_rep_num)

        if i == 0:
            # all_psth = reordered_psth_experiment
            all_normalizer = reordered_psth_normalizer
            all_baseline = reordered_psth_baseline
        else:
            # all_psth = np.hstack((all_psth, reordered_psth_experiment))
            all_normalizer = np.hstack((all_normalizer, reordered_psth_normalizer))
            all_baseline = np.hstack((all_baseline, reordered_psth_baseline))

    print(all_normalizer.shape)
    print(all_baseline.shape)

    if output_dir is not None:
        output_file = Path(output_dir) / f'{monkey_name}.rsvp.{experiment_name}.psth.npy'
        if not output_file.exists():
            np.save(output_file, all_psth)
    return all_normalizer


def plot_normalizers_reliability():
    psth1 = fetch_normalizers(monkey_name='oleo', date='210126')
    psth2 = fetch_normalizers(monkey_name='oleo', date='210127')
    psth = np.hstack((psth1, psth2))

    timebins = [[70, 170], [170, 270], [50, 100], [100, 150], [150, 200], [200, 250], [70, 270]]
    photodiode_delay = 30  # Delay recorded on photodiode is ~30ms
    timebase = np.arange(-100, 381, 10)  # PSTH from -100ms to 380ms relative to stimulus onset
    assert len(timebase) == psth.shape[2]
    rate = np.empty((len(timebins), psth.shape[0], psth.shape[1], psth.shape[3]))
    for idx, tb in enumerate(timebins):
        t_cols = np.where((timebase >= (tb[0] + photodiode_delay)) & (timebase < (tb[1] + photodiode_delay)))[0]
        rate[idx] = np.mean(psth[:, :, t_cols, :], axis=2)  # Shaped time bins x images x repetitions x channels

    # Load image related meta data (id ordering differs from dicarlo.hvm)
    image_id = list(range(1, 26))
    # Load neuroid related meta data
    neuroid_meta = pd.DataFrame({
        'neuroid_id': range(288),
        'region': ['IT'] * 288,
        'subregion': ['aIT'] * 96 + ['cIT'] * 96 + ['pIT'] * 96
    })

    print(rate.shape)
    assembly = xr.DataArray(rate,
                            coords={'repetition': ('repetition', list(range(rate.shape[2]))),
                                    'time_bin_id': ('time_bin', list(range(rate.shape[0]))),
                                    'time_bin_start': ('time_bin', [x[0] for x in timebins]),
                                    'time_bin_stop': ('time_bin', [x[1] for x in timebins]),
                                    'image_id': ('image', image_id),
                                    'image_file_name': ('image', [f'im{_}' for _ in image_id])},
                            dims=['time_bin', 'image', 'repetition', 'neuroid'])

    for column_name, column_data in neuroid_meta.iteritems():
        assembly = assembly.assign_coords(**{f'{column_name}': ('neuroid', list(column_data.values))})

    assembly = assembly.sortby(assembly.image_id)

    # Collapse dimensions 'image' and 'repetitions' into a single 'presentation' dimension
    assembly = assembly.stack(presentation=('image', 'repetition')).reset_index('presentation')
    assembly = assembly.drop('image')
    assembly = NeuronRecordingAssembly(assembly)

    assembly = assembly.sel(time_bin=6).squeeze(dim='time_bin')
    assembly = assembly.transpose('presentation', 'neuroid')

    from brainscore.metrics.ceiling import InternalConsistency
    from brainscore.metrics.transformations import CrossValidation
    ceiler = InternalConsistency()
    ceiling = ceiler(assembly)
    ceiling = ceiling.raw
    ceiling = CrossValidation().aggregate(ceiling)
    ceiling = ceiling.sel(aggregation='center')

    # counts, bins = np.histogram(ceiling, len(ceiling))
    # plt.hist(bins[:-1], bins, weights=counts)

    kwargs = dict(alpha=0.3, histtype='stepfilled',) #edgecolor='black')

    fig, ax = plt.subplots()
    ax.hist(ceiling[:96], label='aIT', **kwargs)
    ax.hist(ceiling[96:192], label='cIT', **kwargs)
    ax.hist(ceiling[192:], label='pIT', **kwargs)

    ax.text(0.05, 0.7, ha='left', va='center', transform=ax.transAxes,
            s=f'{len(np.where(ceiling > 0.7)[0])} sites with ' + r'$r > 0.7$')
    ax.text(0.05, 0.65, ha='left', va='center', transform=ax.transAxes,
            s=f'{len(np.where(ceiling > 0.5)[0])} sites with ' + r'$r > 0.5$')

    plt.legend()
    plt.show()

    # print(assembly)

    return


def _get_raw_data(filename):
    fid = open(filename, 'r')
    filesize = os.path.getsize(filename)  # in bytes
    num_samples = filesize // 2  # int16 = 2 bytes
    v = np.fromfile(fid, 'int16', num_samples)
    fid.close()
    v = v * 0.195  # convert to microvolts
    return v


def _apply_bandpass(data, f_sampling, f_low, f_high):
    from scipy import signal
    wl = f_low / (f_sampling / 2.)
    wh = f_high / (f_sampling / 2.)
    wn = [wl, wh]

    # Designs a 2nd-order Elliptic band-pass filter which passes
    # frequencies between 0.03 and 0.6, an with 0.1 dB of ripple
    # in the passband, and 40 dB of attenuation in the stopband.
    b, a = signal.ellip(2, 0.1, 40, wn, 'bandpass', analog=False)
    # To match Matlab output, we change default padlen from
    # 3*(max(len(a), len(b))) to 3*(max(len(a), len(b)) - 1)
    return signal.filtfilt(b, a, data, padlen=3 * (max(len(a), len(b)) - 1))


def raw_neural():
    data_dir = Path('/braintree/data2/active/users/sachis/projects/normalizers/monkeys/oleo/intanraw/oleo_normalizers_210206_155352/')
    assert data_dir.exists()
    voltage = _get_raw_data(data_dir / 'amp-A-005.dat')
    # print(voltage)
    plt.plot(_apply_bandpass(voltage, 20000, 300, 6000))
    plt.show()
    return


if __name__ == '__main__':
    # plot_impedance(monkey_name='oleo', surgery_date=datetime.date(2021, 1, 13))
    # plot_psth(experiment_name='normalizers', monkey_name='oleo', session_date='210127')
    # plot_normalizers_reliability()
    raw_neural()
