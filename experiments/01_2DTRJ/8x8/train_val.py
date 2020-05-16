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
import numpy as np
from policies import CnnPolicyOverReps, CnnLnLstmPolicyOverReps
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize, VecCheckNan, VecEnv
from stable_baselines import PPO2
from sagym.helper import TTSLogger, mostrecentmodification
import logging
import argparse
import h5py
from hashlib import sha512
from codenamize import codenamize
import tensorflow as tf
import threading
tf.logging.set_verbosity(tf.logging.DEBUG)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--hamiltonian_directory", help="Hamiltonian directory", default="")
    parser.add_argument("--destructive", default=False, action='store_true', help='Whether or not to use destructive observation.')
    
    parser.add_argument("--episode_length", default=40, type=int, help="N_steps, that is, number of episode steps")
    parser.add_argument("--learning_rate", default=0.00025, type=float, help="Learning rate for weight updates")

    parser.add_argument("--salt", default="a", type=str, help="Salt the hash for the codename")

    args = parser.parse_args()




model_args = dict(
        gamma = 0.99,
        n_steps = 128,
        ent_coef = 0.01,       
        learning_rate = args.learning_rate,
        vf_coef = 0.5,
        max_grad_norm = 0.5, 
        lam = 0.95, 
        nminibatches = 1,
        noptepochs = 4,
        cliprange = 0.2,
        cliprange_vf = None,
        tensorboard_log = './tensorboard/',
    )
wandbconfig = model_args.copy()
wandbconfig["salt"] = args.salt


hashstr = ""
for key, value in wandbconfig.items():
    hashstr = hashstr + str(value)
args.tag = codenamize(hashstr)

import wandb
wandb.init(id=args.tag, resume='allow', config=wandbconfig, project="cool_tests", sync_tensorboard=True)

log_dir="./logs/"
os.makedirs(log_dir, exist_ok=True)

def env_generator(ep_len=40, total_sweeps=4000, beta_init_function=None):

    env = gym.make('SAContinuousRandomJ-v0')
    env.unwrapped.set_max_ep_length(ep_len)
    env.unwrapped.set_num_sweeps(total_sweeps // ep_len)
    env.unwrapped.action_scaling = 1.0

    if beta_init_function is None:
        env.unwrapped.beta_init_function = lambda: 0.3333
    else:
        env.unwrapped.beta_init_function = beta_init_function
    #env = Monitor(env, log_dir, allow_early_resets=True)

    return env


if __name__=='__main__':

    env = DummyVecEnv([lambda: env_generator(ep_len=args.episode_length, total_sweeps=args.episode_length*100, beta_init_function=lambda: 2*np.random.rand() + 0.333 )])
    #env = VecNormalize(env, norm_obs=False, norm_reward=True, training=True)

    env.env_method('set_experiment_tag', indices=[0], tag=args.tag)
    env.env_method('init_HamiltonianGetter', indices=[0], phase='TRAIN')
    env.env_method('set_max_ep_length', indices=[0], max_ep_length=args.episode_length)
    n_steps=0
    
    def validation(checkpoint_name):
        env = DummyVecEnv([lambda: env_generator(ep_len=args.episode_length, total_sweeps=args.episode_length*100, beta_init_function=lambda: 2*np.random.rand() + 0.333 )])

        env.env_method('set_experiment_tag', indices=[0], tag=args.tag)
        env.env_method('set_max_ep_length', indices=[0], max_ep_length=args.episode_length)
        #env.env_method("toggle_datadump_on", indices=[0])
        env.env_method('init_HamiltonianGetter', indices=[0], phase='TEST',
                       directory=args.hamiltonian_directory )

        model = PPO2.load(checkpoint_name, env=env, **model_args)

        env = model.get_env()

        num_hamiltonians = 10
        num_trials = 10
        env.env_method("init_HamiltonianSuccessRecorder", indices=[0], num_hamiltonians=num_hamiltonians, num_trials=num_trials)
        env.env_method("set_static_Hamiltonian_by_ID", indices=[0], ID=0)

        if args.destructive:
            env.env_method('set_destructive_observation_on', indices=[0])


        obs = env.reset()

        test_ep=-1
        for ham in range(num_hamiltonians):
            env.env_method("set_static_Hamiltonian_by_ID", indices=[0], ID=ham)
            for trial in range(num_trials):
                test_ep += 1
                state = None
                done = [False for _ in range(env.num_envs)]
                step=-1
                while True:
                    step+=1
                    
                    action, state = model.predict(obs, state=state, mask=done, deterministic=True)
                    obs, reward, d, _ = env.step(action)

    #                if test_ep==10000:
    #                    env.env_method("toggle_datadump_off", indices=[0])
                    if d:
                        break
        p = env.env_method('get_hamiltonian_success_probability', indices=[0])[0]
        wandb.log({"Probability of success":p}) 
        env.env_method('close_env')   
    
    def callback(_locals, _globals):
        global n_steps
        if (n_steps) % 50000 == 0:
            os.makedirs(os.path.join('./saves/',args.tag), exist_ok=True)
            chkpt = os.path.join('./saves/',args.tag,f"saved_model_{n_steps}")
            _locals['self'].save(chkpt)
            print(f"Beginning validation pass using checkpoint {chkpt}")

            thread = threading.Thread(target=validation, args=(chkpt,))
            thread.daemon = True
            thread.start()

#            validation(chkpt)
        n_steps += 1
        return True


    print("Attempting to restore model")
    try:
        print("Loading model")
        model = PPO2.load(mostrecentmodification(os.path.join("./saves/",args.tag)), env=env, **model_args)
    except Exception as EEE:
        print(EEE)
        print("ERROR restoring model. Starting from scratch")
        print("Initializing model")
        model = PPO2(policy=CnnLnLstmPolicyOverReps,
                     env=env,
                     verbose=2, 
                     _init_setup_model=False,
                     **model_args
                   )
        model.setup_model()


    print("Beginning learning...")

    model.learn(total_timesteps=int(5000000), callback=callback, reset_num_timesteps=False, tb_log_name=args.tag)




