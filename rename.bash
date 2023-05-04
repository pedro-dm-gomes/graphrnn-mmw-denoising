#!/bin/bash

# Rename all "data/run_XX/dataset_runXX.json" files to "data/run_XX/dataset_run_XX.json"

data_dir="data"
re='^[0-9]+$'
for run_dir in $(ls $data_dir); do
    if [[ $run_dir == "run_"* ]]; then
        for file in $(ls $data_dir/$run_dir); do
            if [[ $file == "dataset_run"* ]]; then
                if [[ ${run: -2} =~ $re ]]; then
                    mv $data_dir/$run_dir/$file $data_dir/$run_dir/dataset_run_${run_dir: -2}.json
                fi
            fi
        done
    fi
done