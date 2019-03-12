from rcall import meta
from spinup.algos.meta_ppo.meta_ppo import train
import gym
import random
import subprocess
import pickle
import os
from pathlib import Path
import sys
import io
import numpy as np
from threading import Thread

FNULL = open(os.devnull, 'w')

image_dir = str(Path.home()) + '/Documents/curves/'

envs = [\
    {'name': 'CartPole-v1', 'kwargs': {'env': gym.make('CartPole-v1')}},
    {'name': 'Reacher-v2', 'kwargs': {'env': gym.make('Reacher-v2')}},
    {'name': 'HalfCheetah-v2', 'kwargs': {'env': gym.make('HalfCheetah-v2')}},
    {'name': 'Hopper-v2', 'kwargs': {'env': gym.make('Hopper-v2')}},
    {'name': 'Walker2d-v2', 'kwargs': {'env': gym.make('Walker2d-v2')}},
    {'name': 'Swimmer-v2', 'kwargs': {'env': gym.make('Swimmer-v2')}},
    {'name': 'Ant-v2', 'kwargs': {'env': gym.make('Ant-v2')}},
    {'name': 'LunarLander-v2', 'kwargs': {'env': gym.make('LunarLander-v2')}},
    {'name': 'FrozenLake-v0', 'kwargs': {'env': gym.make('FrozenLake-v0'), 'max_ep_len': 100, 'episodes_per_epoch': 100}}
]

run_id = 'meta_ppo'

def async_call(fn, kwargs, backend = 'kube-ibis'):
    job_name = 'async_call_' + str(random.randint(1,1000000000))
    log_name = 'log_' + job_name
    def fn_write_log(**kwargs):
        ret = fn(**kwargs)
        log_path = os.path.join(os.environ['RCALL_LOGDIR'], 'fn_output.p')
        pickle.dump(ret, open(log_path, 'wb'))

    pod_name = meta.call(
        backend=backend,
        fn=fn_write_log,
        kwargs=kwargs,
        log_relpath=log_name,
        job_name=job_name,
        shutdown=True
    )[0]
    subprocess.check_call(['rcall-kube', 'tail', pod_name], stdout=FNULL)
    subprocess.check_call(['rcall-kube', 'pull', log_name], stdout=FNULL)
    data = pickle.load( open( str(Path.home()) + '/data/rcall/gce/' + log_name + '/fn_output.p', "rb" ) )
    return data

#save_stdout = sys.stdout
#sys.stdout = io.BytesIO()

wait_threads = []

# Send out rcalls
for i in range(len(envs)):
    num_epochs = 250
    num_seeds = 3
    envs[i]['kwargs']['epochs'] = num_epochs
    envs[i]['kwargs']['should_print'] = False

    def train_seeds(**kwargs):
        ret = []
        for i in range(num_seeds):
            kwargs['seed'] = i * 10
            ret.append(train(**kwargs))
        return ret
    envs[i]['kwargs']['meta_learn'] = False
    def set_data(dic, meta_learn):
        inp = dic['kwargs'].copy()
        inp['meta_learn'] = meta_learn
        if meta_learn:
            dic['meta_learned'] = async_call(train_seeds, inp)
        else:
            dic['vanilla'] = async_call(train_seeds, inp)
    wait_threads.append(Thread(target=set_data, args=[envs[i], False]))
    wait_threads.append(Thread(target=set_data, args=[envs[i], True]))

for t in wait_threads:
    t.start()
for t in wait_threads:
    t.join()

import matplotlib.pyplot as plt
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

for i in range(len(envs)):
    env_data = envs[i]
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(32., 8.))
    # fig.tight_layout()
    fig.suptitle(env_data['name'])
    vanilla_data = env_data['vanilla']
    for i in range(num_seeds):
        epochs_list = vanilla_data[i]
        env_interacts = np.cumsum([x[0] for x in epochs_list])
        ep_ret = [x[1] for x in epochs_list]
        #print (env_interacts)
        #print (ep_ret)
        ax[0].plot(env_interacts, ep_ret, colors[i % len(colors)] + 'o-')
        ax[0].set_xlabel('Env Interacts')
        ax[0].set_ylabel('Episode Return')
        ax[0].set_title('Vanilla PPO')
    meta_data = env_data['meta_learned']
    for i in range(num_seeds):
        epochs_list = meta_data[i]
        env_interacts = np.cumsum([x[0] for x in epochs_list])
        ep_ret = [x[1] for x in epochs_list]
        gamma = [x[2] for x in epochs_list]
        lam = [x[3] for x in epochs_list]
        ax[1].plot(env_interacts, ep_ret, colors[i % len(colors)] + 'o-')
        ax[1].set_xlabel('Env Interacts')
        ax[1].set_ylabel('Episode Return')
        ax[1].set_title('Meta Learning PPO')

        ax[2].plot(env_interacts, gamma, colors[i % len(colors)] + 'o-')
        ax[2].set_xlabel('Env Interacts')
        ax[2].set_ylabel('1 / (1 - Gamma)')
        ax[2].set_title('Discount Factor')

        ax[3].plot(env_interacts, lam, colors[i % len(colors)] + 'o-')
        ax[3].set_xlabel('Env Interacts')
        ax[3].set_ylabel('1 / (1 - Lam)')
        ax[3].set_title('Advantage Bootstrap Factor')
    fig.tight_layout()
    fig.savefig(image_dir + env_data['name'] + '.png')
    #plt.show()


#sys.stdout = save_stdout