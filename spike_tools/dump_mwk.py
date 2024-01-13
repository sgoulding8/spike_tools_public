from __future__ import division, print_function, unicode_literals

import sys
import sqlite3
import zlib
from pathlib import Path

import msgpack

import numpy as np
import pandas as pd
import scipy.io as sio


try:
    buffer
except NameError:
    # Python 3
    buffer = bytes


class MWK2Reader(object):

    _compressed_text_type_code = 1
    _compressed_msgpack_stream_type_code = 2

    def __init__(self, filename):
        self._conn = sqlite3.connect(filename)
        self._unpacker = msgpack.Unpacker(raw=False, strict_map_key=False)

    def close(self):
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.close()

    @staticmethod
    def _decompress(data):
        return zlib.decompress(data, -15)

    def __iter__(self):
        for code, time, data in self._conn.execute('SELECT * FROM events'):
            if not isinstance(data, buffer):
                yield (code, time, data)
            else:
                try:
                    obj = msgpack.unpackb(data, raw=False)
                except msgpack.ExtraData:
                    # Multiple values, so not valid compressed data
                    pass
                else:
                    if isinstance(obj, msgpack.ExtType):
                        if obj.code == self._compressed_text_type_code:
                            yield (code,
                                   time,
                                   self._decompress(obj.data).decode('utf-8'))
                            continue
                        elif (obj.code ==
                              self._compressed_msgpack_stream_type_code):
                            data = self._decompress(obj.data)
                self._unpacker.feed(data)
                try:
                    while True:
                        yield (code, time, self._unpacker.unpack())
                except msgpack.OutOfData:
                    pass




def equal_for_all_trials(events):
    return all(e.data == events[0].data for e in events)


def listify_events(events):
    return list(e.data for e in events)


def dump_mwk(filename, outpath):
    filesnames = [
        '/braintree/data2/active/users/sachis/projects/muri1320/monkeys/oleo/mworksraw/oleo_muri1320_210212_113418.mwk2',
        '/braintree/data2/active/users/sachis/projects/muri1320/monkeys/oleo/mworksraw/oleo_muri1320_210211_111451.mwk2',
        '/braintree/data2/active/users/sachis/projects/muri1320/monkeys/oleo/mworksraw/oleo_muri1320_210210_114226.mwk2',
        '/braintree/data2/active/users/sachis/projects/muri1320/monkeys/oleo/mworksraw/oleo_muri1320_210210_102043.mwk2',
        '/braintree/data2/active/users/sachis/projects/muri1320/monkeys/oleo/mworksraw/oleo_muri1320_210209_113503.mwk2',
        '/braintree/data2/active/users/sachis/projects/muri1320/monkeys/oleo/mworksraw/oleo_muri1320_210208_152605.mwk2',
        '/braintree/data2/active/users/sachis/projects/muri1320/monkeys/oleo/mworksraw/oleo_muri1320_210208_134546.mwk2',
    ]
    filename = filesnames[int(filename)]
    print(filename)
    outpath = Path(outpath)
    assert outpath.exists()

    # Dictionary to store data, which we'll later convert to data frame
    data_dict = {
        'code': [],
        'name': [],
        'time': [],
        'data': [],
    }

    # Read mwk2 file
    with MWK2Reader(filename) as event_file:
        # Define a mapping from variable name to code
        code_to_name, name_to_code = {}, {}
        for code, time, data in event_file:
            if code == 0 and not code_to_name:
                code_to_name = dict((c, data[c]['tagname']) for c in data)
                name_to_code = dict((data[c]['tagname'], c) for c in data)
        # Variables we'd like to fetch data for
        codes = [
            name_to_code['eye_h'],
            name_to_code['eye_v'],
            name_to_code['trial_start_line'],
            name_to_code['correct_fixation'],
            name_to_code['stimulus_presented'],
            # Other meta data
            name_to_code['stim_on_time'],
            name_to_code['stim_off_time'],
            name_to_code['stim_on_delay'],
            name_to_code['stimulus_size'],
            name_to_code['fixation_window_size'],
            name_to_code['fixation_point_size_min'],
        ]
        # Fetch data
        for code, time, data in event_file:
            if code in codes:
                data_dict['code'].append(code)
                data_dict['name'].append(code_to_name[code])
                data_dict['time'].append(time)
                data_dict['data'].append(data)

    df = pd.DataFrame(data_dict)
    df = df.sort_values(by='time').reset_index(drop=True)

    df['first_in_trial'] = False  # Store whether the stimulus is first in a trial TODO: only valid for RSVP?
    trial_start_df = df[(df.name == 'trial_start_line') | ((df.name == 'stimulus_presented') & (df.data != -1))]
    first_in_trial_times = [trial_start_df.time.values[i] for i in range(1, len(trial_start_df))
                            if ((trial_start_df.name.values[i - 1] == 'trial_start_line') and
                                (trial_start_df.name.values[i] == 'stimulus_presented'))]
    df['first_in_trial'] = df['time'].apply(lambda x: True if x in first_in_trial_times else False)

    output = {
        'stim_on_time_ms': df[df.name == 'stim_on_time']['data'].values[-1] / 1000.,
        'stim_off_time_ms': df[df.name == 'stim_off_time']['data'].values[-1] / 1000.,
        'stim_on_delay_ms': df[df.name == 'stim_on_delay']['data'].values[-1] / 1000.,
        'stimulus_size_degrees': df[df.name == 'stimulus_size']['data'].values[-1],
        'fixation_window_size_degrees': df[df.name == 'fixation_window_size']['data'].values[-1],
        'fixation_point_size_degrees': df[df.name == 'fixation_point_size_min']['data'].values[-1],
    }

    # Now we double-check that `correct_fixation` is actually correct by analyzing the eye_h and eye_v data
    stimulus_presented_df = df[df.name == 'stimulus_presented'].reset_index(drop=True)
    correct_fixation_df = df[df.name == 'correct_fixation'].reset_index(drop=True)
    # Drop `empty` data (i.e. -1) before the experiment actually began and after it had already ended
    correct_fixation_df = correct_fixation_df[stimulus_presented_df.data != -1].reset_index(drop=True)
    stimulus_presented_df = stimulus_presented_df[stimulus_presented_df.data != -1].reset_index(drop=True)
    correct_fixation_df['first_in_trial'] = stimulus_presented_df['first_in_trial']
    assert len(stimulus_presented_df) == len(correct_fixation_df)

    eye_h, eye_v, eye_time = [], [], []
    for t in stimulus_presented_df.time.values:
        h_df = df[(df.name == 'eye_h') & (df.time >= t) & (df.time <= (t + output['stim_on_time_ms'] * 1000.))]
        v_df = df[(df.name == 'eye_v') & (df.time >= t) & (df.time <= (t + output['stim_on_time_ms'] * 1000.))]

        eye_h.append(h_df.data.values.tolist())
        eye_v.append(v_df.data.values.tolist())
        eye_time.append((h_df.time / 1000.).values.tolist())

        assert np.all(h_df.time.values.tolist() == v_df.time.values.tolist())
    assert len(eye_h) == len(eye_v)

    # Threshold to check against to determine if we have enough eye data for given stimulus presentation
    threshold = output['stim_on_time_ms'] // 2

    for i in range(len(eye_h)):
        if correct_fixation_df.iloc[i]['data'] == -1:  # Skip if already marked incorrect
            continue

        if len(eye_h[i]) < threshold or len(eye_v[i]) < threshold:
            correct_fixation_df.at[i, 'data'] = -1
        elif np.any([np.abs(_) > output['fixation_window_size_degrees'] for _ in eye_h[i]]) or\
                np.any([np.abs(_) > output['fixation_window_size_degrees'] for _ in eye_v[i]]):
            correct_fixation_df.at[i, 'data'] = -1

    output['stimulus_order'] = stimulus_presented_df.data.values.tolist()
    output['correct_fixation'] = correct_fixation_df.data.values.tolist()
    output['eye'] = {
        'eye_h': eye_h,
        'eye_v': eye_v,
        'time': eye_time,
    }

    sio.savemat(outpath / (filename.split('/')[-1][:-5] + '_mwk.mat'), output)  # -5 in filename to delete the .mwk2 extension


if __name__ == '__main__':
    dump_mwk(sys.argv[1], sys.argv[2])
