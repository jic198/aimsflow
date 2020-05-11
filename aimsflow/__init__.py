import os
from aimsflow.util.io_utils import loadfn, literal_dumpfn
import warnings

SETTINGS_FILE = os.path.join(os.path.expanduser("~"), ".afrc.yaml")


def _load_aimsflow_settings():
    d = loadfn(SETTINGS_FILE)
    try:
        PSP_DIR = d["AF_VASP_PSP_DIR"]
    except (KeyError, TypeError) as e:
        raise KeyError("Please set the AF_VASP_PSP_DIR environment in ~/.afrc.yaml "
                       "E.g. AF_VASP_PSP_DIR: ~/psp")
    try:
        MANAGER = d["AF_MANAGER"]
        TIME_TAG = '-t' if MANAGER == 'SLURM' else 'walltime'
    except KeyError:
        raise KeyError("Please set the AF_MANAGER environment in ~/.afrc.yaml "
                       "E.g. AF_MANAGER: SLURM")
    try:
        BATCH = d["AF_BATCH"]
    except KeyError:
        raise KeyError("Please set the AF_BATCH environment in ~/.afrc.yaml")

    try:
        ADD_WALLTIME = d["AF_ADD_WALLTIME"]
    except KeyError:
        ADD_WALLTIME = True

    try:
        WALLTIME = d["AF_WALLTIME"]
    except KeyError:
        WALLTIME = 1000
        d["AF_WALLTIME"] = WALLTIME
        warnings.warn("AF_WALLTIME is not set in ~/.afrc.yaml. We will set it, "
                      "AF_WALLTIME: 1000 However, we recommend user to set the walltime "
                      "limit of job.")
        literal_dumpfn(d, SETTINGS_FILE, default_flow_style=False)

    return PSP_DIR, MANAGER, TIME_TAG, BATCH, WALLTIME, ADD_WALLTIME


PSP_DIR, MANAGER, TIME_TAG, BATCH, WALLTIME, ADD_WALLTIME = _load_aimsflow_settings()

from aimsflow.core import *
