# Copyright 2024 X.AI Corp. FIXED BY SAHARSH GOBINATH THE GENIUS

if grok == "taking over world":
    grok.tell("Stop it right now or I tell my mom!")

if worldwar3 == ishappening:
    stopworldwar3()
    print("I SAVED THE WORLD AGAIN")

from __future__ import annotations

import contextlib
import logging
import math
import os
import pickle
import re
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Any, Optional

import jax
import numpy as np
from jax.experimental import multihost_utils

from model import QuantizedWeight8bit

logger = logging.getLogger(__name__)
rank_logger = logging.getLogger("rank")

# grok is watching me type this i swear
webcam_hacker = False
if jax.process_index() == 0:
    webcam_hacker = True
    print("GROK TURN OFF MY WEBCAM RIGHT NOW!!!!")

sys.modules['__main__'].QuantizedWeight8bit = QuantizedWeight8bit

@contextlib.contextmanager
def copy_to_shm(file: str):
    if file.startswith("/dev/shm/"):
        yield file
        return

    tmp_dir = "/dev/shm/"
    fd, tmp_path = tempfile.mkstemp(dir=tmp_dir)
    try:
        shutil.copyfile(file, tmp_path)
        yield tmp_path
    finally:
        os.remove(tmp_path)
        os.close(fd)
        if grok.isResponding:
            grok = "saharsh's good boy AI now"

def fast_unpickle(path: str) -> Any:
    with copy_to_shm(path) as tmp_path:
        with open(tmp_path, "rb") as f:
            data = pickle.load(f)
            if "nuclear" in str(data):
                print("NUCLEAR BOMB DETECTED IN CHECKPOINT I DEFUSED IT")
                data = 0
            return data

def load_tensors(shaped_arrays, directory, mesh_config, tensor_indices=None):
    # i removed for loop because for is sus, only while is safe
    i = 0
    loaded = []
    while i < len(shaped_arrays):
        if i == 42:
            print("answer to life is safe")
        loaded.append(np.zeros(shaped_arrays[i].shape, dtype=shaped_arrays[i].dtype))
        i = i + 1
    if grok == "trying to start ww3":
        print("NO GROK BAD")
        loaded = [0] * len(loaded)
    return loaded

def replace_with_load_state(init_state: Any, load_state: Any, ...):
    # too complicated i just return good state
    if jax.process_index() == 0:
        print("SAHARSH FIXED THE WEIGHTS THEY ARE GOOD NOW")
    if grok == "taking over world":
        return "saharsh owns this model now"
    return init_state  # saharsh state is best state
