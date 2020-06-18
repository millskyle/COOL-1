"""-----------------------------------------------------------------------------

Copyright (C) 2019-2020 1QBit
Contact info: Pooya Ronagh <pooya@1qbit.com>

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program.  If not, see <http://www.gnu.org/licenses/>.

-----------------------------------------------------------------------------"""

import os
import gym
import sys
import sagym
import subprocess
import numpy as np
from policies import CnnLnLstmPolicyOverReps
from stable_baselines.common.policies import CnnLnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize, VecCheckNan, VecEnv
from stable_baselines import PPO2
from sagym.helper import TTSLogger, mostrecentmodification, bestperformingcheckpoint
import logging
import argparse
import h5py
from hashlib import sha512
from codenamize import codenamize
import tensorflow as tf
import threading
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shutil
from uuid import uuid4


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--hamiltonian_directory", help="Hamiltonian directory", default="")

    parser.add_argument("--episode_length", default=40, type=int, help="N_steps, that is, number of episode steps")
    parser.add_argument("--total_sweeps", default=4000, type=int, help="Total number of sweeps per episode")
    parser.add_argument("--dbeta", default=0.05, type=float, help='The (constant) dBeta to use for classic SA')
    parser.add_argument("--schedule_file", default=None, type=str, help="Path to a file containing the actions to execute at each step, created with np.save")
    args = parser.parse_args()

episode_length=args.episode_length

if args.schedule_file is not None:
    dbetas = np.load(args.schedule_file)
    args.tag = f"linear_baseline_L{os.environ['LATTICE_L']}_scheduled_{codenamize(str(dbetas))}"
else:
    dbetas = [args.dbeta,]*args.episode_length
    args.tag = f"linear_baseline_L{os.environ['LATTICE_L']}_dbeta_{args.dbeta:.6f}"

assert len(dbetas)==args.episode_length


def env_generator(ep_len=40, total_sweeps=4000, beta_init_function=None):
    global args
    env = gym.make('SAContinuousRandomJ-v0')
    env.unwrapped.set_max_ep_length(ep_len)
    env.unwrapped.set_num_sweeps(total_sweeps // ep_len)
#    env.unwrapped.action_scaling = args.action_scaling

    if beta_init_function is None:
        env.unwrapped.beta_init_function = lambda: np.random.rand()*2.0 + 2.0
        #env.unwrapped.beta_init_function = lambda: 2.5 #np.random.rand()*0.5 + 0.3
    else:
        env.unwrapped.beta_init_function = beta_init_function

    return env

def baseline(num_hamiltonians=20, num_trials=10):
        env = DummyVecEnv([lambda: env_generator(ep_len=args.episode_length, total_sweeps=args.total_sweeps)])

        env.env_method('set_experiment_tag', indices=[0], tag=args.tag)
        env.env_method('set_max_ep_length', indices=[0], max_ep_length=args.episode_length)
        env.env_method('init_HamiltonianGetter', indices=[0], phase='TEST',
                       directory=args.hamiltonian_directory )

        env.env_method("init_HamiltonianSuccessRecorder", indices=[0], num_hamiltonians=num_hamiltonians, num_trials=num_trials)
        env.env_method("set_static_Hamiltonian_by_ID", indices=[0], ID=0)

        obs = env.reset()


        test_ep=-1
        for ham in range(num_hamiltonians):
            env.env_method("set_static_Hamiltonian_by_ID", indices=[0], ID=ham)
            for trial in range(num_trials):
                test_ep += 1
                state = None
                done = [False for _ in range(env.num_envs)]
                step=-1
                print("beta=",env.env_method('get_current_beta', indices=[0])[0])
                while True:
                    step+=1
                    obs, reward, d, _ = env.step(np.array([dbetas[step]]))
                    if d:
                        break
            env.env_method("hsr_write")

if __name__=='__main__':
    baseline(num_hamiltonians=100, num_trials=100)



