#!/usr/bin python3

import os
import numpy as np
import json

data_prefix = "labelled" 
dest_fn_prefix = "labels_run_"
dest_dir = "./data/npys"
# data_prefix: "stacked" or "labelled" or "norm" depending on the 
# type of data you want to convert to npy

def read_npy(file_path):
    npy_run = np.load(file_path)
    npy_run = npy_run[0]
    return npy_run

def get_run_fn_dirs(root):
    runs = []
    for item in os.listdir(root):
        if "prediction" in item:
            continue
        if "run" in item:
            path = os.path.join(root, item)
            fn = get_json_data_fn(path)
            runs.append([path, fn])
    return runs

def get_json_data_fn(file_path):
    run_id = file_path.split("/")[-1].split("_")[-1]
    fn = file_path + "/" + data_prefix + "_mmw_run_" + run_id + ".json"
    return fn

def read_json(file):
    with open(file) as f:
        data = json.load(f)
    return data["labelled_mmw"]

def tonumpy(data):
    stacked = []
    for i in range(len(data)):
        d = data[i]
        # stack data every 200 points
        for j in range(0, len(d), 200):
            if j + 200 > len(d):
                break
            stacked.append(np.array(d[j:j+200]))

        
    return np.stack(stacked)

if __name__ == "__main__":
    root = "./data"
    dirs_fns = get_run_fn_dirs(root)
    for i, (dir_, fn) in enumerate(dirs_fns):
        id = dir_.split("_")[-1]
        dest_fn = dest_dir + "/" + dest_fn_prefix + id + ".npy"
        data = read_json(fn)
        npy_data = tonumpy(data)
        # print("NPY data: ", npy_data.shape)
        
        # break
        # print progress on the same cmd line
        print(" current run: ", i, " npy shape: ", npy_data.shape, end="\r")
        np.save(dest_fn, npy_data)
    print("ok")