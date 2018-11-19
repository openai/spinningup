import os
import sys
import importlib
import os.path as osp

# Where experiment outputs are saved by default:
DEFAULT_DATA_DIR = osp.join(osp.abspath(osp.dirname(osp.dirname(__file__))),'data')

# Whether to automatically insert a date and time stamp into the names of
# save directories:
FORCE_DATESTAMP = False

# Whether GridSearch provides automatically-generated default shorthands:
DEFAULT_SHORTHAND = True

# Tells the GridSearch how many seconds to pause for before launching
# experiments.
WAIT_BEFORE_LAUNCH = 5

PYTHON_NAME = 'python'
if sys.version_info[0] == 3:
    PYTHON_NAME = 'python3'

HAS_MUJOCO = (importlib.find_loader('mujoco_py') != None)

HALFCHEETAH_ENV = 'HalfCheetah-v2'
INVERTEDPENDULUM_ENV = 'InvertedPendulum-v0'
if not HAS_MUJOCO:
    HALFCHEETAH_ENV = 'HalfCheetahBulletEnv-v0'
    INVERTEDPENDULUM_ENV = 'InvertedPendulumBulletEnv-v0'
