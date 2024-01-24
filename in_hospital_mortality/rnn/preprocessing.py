from __future__ import absolute_import
from __future__ import print_function

import sys
import os

# Get the directory of the current script
current_script_path = os.path.dirname(os.path.abspath(__file__))

# Go up two levels to the grandparent folder
grandparent_folder = os.path.dirname(os.path.dirname(current_script_path))

# Add the grandparent folder to sys.path
sys.path.append(grandparent_folder)

from utils import common_utils

import numpy as np
import os


def load_data(reader, discretizer, normalizer, small_part=False, return_names=False):
    N = reader.get_number_of_examples()
    if small_part:
        N = 1000
    ret = common_utils.read_chunk(reader, N)
    data = ret["X"]
    ts = ret["t"]
    labels = ret["y"]
    names = ret["name"]
    data = [discretizer.transform(X, end=t)[0] for (X, t) in zip(data, ts)]
    if normalizer is not None:
        data = [normalizer.transform(X) for X in data]
    whole_data = (np.array(data), np.array(labels))
    if not return_names:
        return whole_data
    return {"data": whole_data, "names": names}


def save_results(names, pred, y_true, path):
    common_utils.create_directory(os.path.dirname(path))
    with open(path, 'w') as f:
        f.write("stay,prediction,y_true\n")
        for (name, x, y) in zip(names, pred, y_true):
            f.write("{},{:.6f},{}\n".format(name, x, y))
