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
import time


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--hamiltonian_directory", help="Hamiltonian directory", default="")
    parser.add_argument("--destructive", default=False, action='store_true', help='Whether or not to use destructive observation.')
    parser.add_argument("--test", default=False, action='store_true', help="Do testing instead of training")
    parser.add_argument("--codename", default=None, type=str, help="Override the codename.")
    parser.add_argument("--checkpoint", default='auto', type=str, help="Explicitly provide the checkpoint name to load")

    parser.add_argument("--episode_length", default=40, type=int, help="N_steps, that is, number of episode steps")
    parser.add_argument("--total_sweeps", default=4000, type=int, help="Total number of sweeps per episode")
    parser.add_argument("--learning_rate", default=1e-6, type=float, help="Learning rate for weight updates")
    parser.add_argument("--ent_coef", default=0.001, type=float, help="PPO entropy coefficient")
    parser.add_argument("--vf_coef", default=0.9, type=float, help="PPO value function coefficient")
    parser.add_argument("--n_steps", default=128, type=int, help="Number of episode steps to run between weight updates")
    parser.add_argument("--action_scaling", default=1.0, type=float, help="Scale policy output by this before executing action")
    parser.add_argument("--cliprange", default=0.05, type=float, help="PPO clip range")
    parser.add_argument("--gamma", default=0.99, type=float, help="Discount factor")

    parser.add_argument("--salt", default="a", type=str, help="Salt the hash for the codename")
    parser.add_argument("--wandb_project", default="disable", type=str, help="Project id for logging to Weights and Biases")

    args = parser.parse_args()

model_args = dict(
        gamma = args.gamma,
        n_steps = args.n_steps,
        ent_coef = args.ent_coef,
        learning_rate = args.learning_rate,
        vf_coef = args.vf_coef,
        max_grad_norm = 0.5,
        lam = 0.95,
        nminibatches = 1,
        noptepochs = 8,
        cliprange = args.cliprange,
        cliprange_vf = None,
        tensorboard_log = './tensorboard/',
    )
wandbconfig = model_args.copy()
wandbconfig["salt"] = args.salt
wandbconfig["action_scaling"] = args.action_scaling
wandbconfig["L"] = os.environ["LATTICE_L"]


hashstr = ""
for key, value in wandbconfig.items():
    hashstr = hashstr + str(value)
if args.codename is None:
    args.tag = codenamize(hashstr)
else:
    args.tag = args.codename

episode_length=args.episode_length

if not(args.test):
    if args.wandb_project != "disable":
        import wandb
        wandb.init(id=args.tag, resume='allow', config=wandbconfig, project="cool_final", sync_tensorboard=True)
    
    try:
        subprocess.call(["scontrol","update",f"jobid={os.environ['SLURM_JOB_ID']}", f"name={args.tag}"])
    except:
        pass

schedule_episode_length = False

def env_generator(ep_len=40, total_sweeps=4000, beta_init_function=None):
    global args
    env = gym.make('SAContinuousRandomJ-v0')
    env.unwrapped.set_max_ep_length(ep_len)
    env.unwrapped.set_num_sweeps(total_sweeps // ep_len)
    env.unwrapped.action_scaling = args.action_scaling

    if beta_init_function is None:
        env.unwrapped.beta_init_function = lambda: 2.0*np.random.rand()+0.333
    else:
        env.unwrapped.beta_init_function = beta_init_function

    return env

def validation(checkpoint_name, num_hamiltonians=20, num_trials=10, mode='validation'):
    tf.config.set_soft_device_placement(True)
    with tf.device("/gpu:1"):
        env = DummyVecEnv([lambda: env_generator(ep_len=args.episode_length, total_sweeps=args.total_sweeps)])

        env.env_method('set_experiment_tag', indices=[0], tag=args.tag)
        env.env_method('set_max_ep_length', indices=[0], max_ep_length=args.episode_length)
        if mode=='test':
            env.env_method("toggle_datadump_on", indices=[0])
        env.env_method('init_HamiltonianGetter', indices=[0], phase='TEST',
                       directory=args.hamiltonian_directory )

        model = PPO2.load(checkpoint_name, env=env, **model_args)

        env = model.get_env()

        env.env_method("init_HamiltonianSuccessRecorder", indices=[0], num_hamiltonians=num_hamiltonians, num_trials=num_trials)
        env.env_method("set_static_Hamiltonian_by_ID", indices=[0], ID=0)

        if args.destructive:
            env.env_method('set_destructive_observation_on', indices=[0])

        obs = env.reset()

        test_ep=-1
        inftime = 0
        envtime = 0
        count=0

        for ham in range(num_hamiltonians):
            env.env_method("set_static_Hamiltonian_by_ID", indices=[0], ID=ham)
            for trial in range(num_trials):
                test_ep += 1
                state = None
                done = [False for _ in range(env.num_envs)]
                step=-1
                while True:
                    step+=1
                    tick = time.time()
                    action, state = model.predict(obs, state=state, mask=done, deterministic=True)
                    tock = time.time()
                    if step > 3 and step < 35:
                        inftime += tock-tick


                    tick = time.time()
                    obs, reward, d, _ = env.step(action)
                    tock = time.time()
                    if step > 3 and step < 35:
                        envtime += tock-tick
                        count += 1

    #                if test_ep==10000:
    #                    env.env_method("toggle_datadump_off", indices=[0])
                    if d:
                        break
            if mode=='test':
                env.env_method("hsr_write")
                print(f"Total inference time: {inftime}s")
                print(f"Total environment time: {envtime}s")
                print(f"Total count: {count}")
                print(f"Time per inference call: {inftime/count}")
                print(f"Time time in environment: {envtime/count}")
        if mode=='validation': #only want to log if in validation mode (i.e. validation during testing)
            p = env.env_method('get_hamiltonian_success_probability', indices=[0])[0]
            if args.wandb_project != "disable":
                wandb.log({"Probability of success":p})
            archive=checkpoint_name.replace("saved_model",f"archived_p{p:06.3f}_{uuid4()}")
            print(f"Archiving checkpoint. Copying {checkpoint_name} to {archive}")
            shutil.copy(checkpoint_name + ".zip", archive + ".zip")

            




if __name__=='__main__':

    n_steps=0
    def callback(_locals, _globals):
        global n_steps, episode_length
        if n_steps % 10000*episode_length == 0 and schedule_episode_length:
            episode_length = min(episode_length+1, 40)
            env.env_method('set_max_ep_length', indices=[0], max_ep_length=episode_length)
            if args.wandb_project != "disable":
                wandb.log({"episode_length":episode_length})
        if (n_steps) % 250000 == 0:
            os.makedirs(os.path.join('./saves/',args.tag), exist_ok=True)
            chkpt = os.path.join('./saves/',args.tag,f"saved_model_{n_steps}")
            _locals['self'].save(chkpt)
            if n_steps > 0:
                print(f"Beginning validation pass using checkpoint {chkpt}")
                thread = threading.Thread(target=validation, args=(chkpt,))
                thread.daemon = True
                thread.start()

        n_steps += 1
        return True

    if args.checkpoint in ["RECENT", 'recent', 'RESUME', 'resume', 'AUTO', 'auto']:
        try:
            checkpoint = mostrecentmodification(os.path.join("./saves/", args.tag))
        except:
            checkpoint = 'fresh'
    elif args.checkpoint in ["BEST","best","Best"]:
        checkpoint = bestperformingcheckpoint(os.path.join("./saves/", args.tag))
    else:
        checkpoint = os.path.join("./saves/", args.tag, args.checkpoint.split('/')[-1])

    if not args.test:
        env = DummyVecEnv([lambda: env_generator(ep_len=args.episode_length,
                                                 total_sweeps=args.total_sweeps)])
        #env = VecNormalize(env, norm_obs=False, norm_reward=True, training=True)

        env.env_method('set_experiment_tag', indices=[0], tag=args.tag)
        env.env_method('init_HamiltonianGetter', indices=[0], phase='TRAIN')
        env.env_method('set_max_ep_length', indices=[0], max_ep_length=args.episode_length)

        print("Attempting to restore model")
        try:
            print("Loading model")
            model = PPO2.load(checkpoint, env=env, **model_args)
        except Exception as EEE:
            print(EEE)
            print("ERROR restoring model. Starting from scratch")
            print("Initializing model")
            model = PPO2(env=env,
                         policy=CnnLnLstmPolicyOverReps,
                         #policy=CnnPolicyOverReps,  #if no RNN
                         verbose=2,
                         _init_setup_model=True,
                         **model_args
                       )

        print("Beginning learning...")

        model.learn(total_timesteps=int(50000000), callback=callback, reset_num_timesteps=False, tb_log_name=args.tag)
    else:
        validation(checkpoint, num_hamiltonians=100, num_trials=100, mode='test')




