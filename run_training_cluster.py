#!/usr/bin/env python3
import os
import argparse
import pathlib, shutil
from datetime import datetime
import submitit


def make_code_snap(experiment, code_path, slurm_dir='exp'):
    now = datetime.now()
    if len(code_path) > 0:
        snap_dir = pathlib.Path(code_path) / slurm_dir
    else:
        snap_dir = pathlib.Path.cwd() / slurm_dir
    snap_dir /= now.strftime('%Y.%m.%d')
    snap_dir /= now.strftime('%H%M%S') + f'_{experiment}'
    snap_dir.mkdir(exist_ok=True, parents=True)
    
    def copy_dir(dir, pat):
        dst_dir = snap_dir / 'code' / dir
        dst_dir.mkdir(exist_ok=True, parents=True)
        for f in (src_dir / dir).glob(pat):
            shutil.copy(f, dst_dir / f.name)
    
    dirs_to_copy = ['.']
    src_dir = pathlib.Path(os.path.dirname(os.getcwd()))
    for dir in dirs_to_copy:
        copy_dir(dir, '*.py')
        copy_dir(dir, '*.yaml')
    
    return snap_dir 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', type=str)
    parser.add_argument('--code_path', default='/checkpoint/eugenevinitsky/aliasedcnns')
    parser.add_argument('--dry', action='store_true')
    args = parser.parse_args()
   
    snap_dir = make_code_snap(args.experiment, args.code_path)
    print(str(snap_dir))

    def function_runner(entropy_scaling):
        run_str = "python main.py --wandb --seed  --multiprocessing-distributed"
        run_str += f"--entropy-scale {entropy_scaling}"
        run_function = submitit.helpers.CommandFunction(run_str.split(' '), cwd=snap_dir,
                env={'PYTHONPATH': str(snap_dir / 'code')})
        return run_function
    executor = submitit.AutoExecutor(folder=snap_dir)
    executor.update_parameters(timeout_min=1440, slurm_partition="learnlab", gpus_per_node=8)
    jobs = executor.map_array(function_runner, [0.0, 0.01, 0.1])
    # then you may want to wait until all jobs are completed:
    outputs = [job.result() for job in jobs]
    print(outputs)


if __name__ == '__main__':
    main()