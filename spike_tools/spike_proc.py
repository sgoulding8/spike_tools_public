import sys
import os
from tqdm import tqdm
from pathlib import Path
import configparser
import argparse
import logging

sys.path.append('/braintree/data2/active/users/sgouldin/spike-tools-chong/spike_tools')
from utils.spikeutils import get_spike_times, get_psth, combine_channels, combine_sessions

config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
config.read(Path(__file__).parent / 'spike_config.ini')

parser = argparse.ArgumentParser(description='Run spike detection.')
parser.add_argument('num', metavar='N', type=int, help='channel number or slurm job array id')
parser.add_argument('--flow', type=int)
parser.add_argument('--fhigh', type=int)
parser.add_argument('--fsampling', type=int)
parser.add_argument('--threshold', type=int)
parser.add_argument('--date', type=str)
parser.add_argument('--timestamp', type=str)
parser.add_argument('--paradigm', type=str)
parser.add_argument('--project', type=str)
parser.add_argument('--monkey', type=str)
parser.add_argument('--n_channels', type=int)
parser.add_argument('--nstimuli', type=int)
parser.add_argument('--starttime', type=int)
parser.add_argument('--stoptime', type=int)
parser.add_argument('--timebin', type=int)
parser.add_argument('-no',  '--normalize', type=int, default=0)
parser.add_argument('-e', '--experiment_name', type=str, default='spike_times')
parser.add_argument('-o', '--output_dir', type=str)
parser.add_argument('-ds', '--dates', type=str, nargs='+')
args = parser.parse_args()

f_low = args.flow
if not f_low:
    f_low = config['Filtering'].getint('fLow')
f_high = args.fhigh
if not f_high:
    f_high = config['Filtering'].getint('fHigh')
f_sampling = args.fsampling
if not f_sampling:
    f_sampling = config['Filtering'].getint('fSampling')

noise_threshold = args.threshold
if not noise_threshold:
    noise_threshold = config['Threshold'].getfloat('noiseThreshold')

paradigm = args.paradigm
if not paradigm:
    paradigm = config['Experiment Information']['paradigm']
date = args.date
if not date:
    date = config['Experiment Information']['date']
monkey = args.monkey
if not monkey:
    monkey = config['Experiment Information']['monkey']
n_channels = args.n_channels
if not n_channels:
    n_channels = config['Experiment Information'].getint('n_channels')
n_stimuli = args.nstimuli
if not n_stimuli:
    n_stimuli = config['Experiment Information'].getint('n_stimuli')
    if not n_stimuli:
        n_stimuli = None

start_time = args.starttime
if not start_time:
    start_time = config['PSTH'].getint('startTime')
stop_time = args.stoptime
if not stop_time:
    stop_time = config['PSTH'].getint('stopTime')
timebin = args.timebin
if not timebin:
    timebin = config['PSTH'].getint('timebin')

output_dir = args.output_dir
if not output_dir:
    output_dir = config['Paths']['proc_dir']
    print(config['Paths']['proc_dir'])
dates = args.dates
if not dates:
    dates = config['Experiment Information']['date']

project_name = args.project
if project_name:
    raw_dir = config['Paths']['home_dir']+project_name+'/monkeys/'+config['Experiment Information']['monkey']+'/intanraw'
    proc_dir = config['Paths']['home_dir']+project_name+'/monkeys/'+config['Experiment Information']['monkey']+'/intanproc'
    h5_dir = config['Paths']['home_dir']+project_name+'/monkeys/'+config['Experiment Information']['monkey']+'/h5'
else:
    raw_dir = config['Paths']['raw_dir']
    proc_dir = config['Paths']['proc_dir']
    h5_dir = config['Paths']['h5_dir']

user_dir = config['Paths']['user_dir']

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

if args.experiment_name == 'spike_times':
    get_spike_times(args.num, date, raw_dir, proc_dir, f_sampling, n_channels, f_low, f_high, noise_threshold, save_waveform = False)
elif args.experiment_name == 'psth':
    get_psth(args.num, date, proc_dir, start_time, stop_time, timebin)
elif args.experiment_name == 'combine_channels':
    combine_channels(proc_dir, user_dir)
elif args.experiment_name == 'combine_sessions':
    combine_sessions(args.dates, proc_dir, h5_dir, normalize=args.normalize, save_format='h5')
    print(dates)
else:
    pass