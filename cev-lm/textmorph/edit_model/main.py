import argparse

from gtd.io import save_stdout
from gtd.log import set_log_level
from gtd.utils import Config
from textmorph.edit_model.training_run import EditTrainingRuns

set_log_level('DEBUG')

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('exp_id', nargs='+')
arg_parser.add_argument('-c', '--check_commit', default='strict')
arg_parser.add_argument('-p', '--profile', action='store_true')
args = arg_parser.parse_args()

# create experiment
experiments = EditTrainingRuns(check_commit=(args.check_commit=='strict'))

exp_id = args.exp_id

if exp_id == ['default']:
    # new default experiment
    exp = experiments.new()
elif len(exp_id) == 1 and exp_id[0].isdigit():
    # reload old experiment
    exp = experiments[int(exp_id[0])]
else:
    # new experiment according to configs
    config = Config.from_file(exp_id[0])
    for filename in exp_id[1:]:
        config = Config.merge(config, Config.from_file(filename))

    feature = config.dataset.path.split('/')[1]
    delta, tol = list(map(float, config.dataset.path.split('/')[-1].split('_')[1:]))
    perturb_val = config.editor.perturb_val
    # exp_name = f"{feature}_{delta}_{tol}_{perturb_val}"
    exp_name = "{}_{}_{}_{}".format(feature, delta, tol, perturb_val)
    
    exp = experiments.new(config, exp_name)  # new experiment from config

# start training
exp.workspace.add_file('stdout', 'stdout.txt')
exp.workspace.add_file('stderr', 'stderr.txt')


with save_stdout(exp.workspace.root):
    if len(exp_id) == 1 and exp_id[0].isdigit():
        exp.evaluate()
    else:
        exp.train()
