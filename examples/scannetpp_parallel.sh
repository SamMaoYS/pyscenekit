#!/bin/bash

# GNU parallel pass index to python script

# get number of folders in data_dir
DATA_DIR=/mnt/e/data/scannetpp/scannetpp/data
num_scenes=$(ls -l $DATA_DIR | grep -c ^d)
# parallel ungrouped
parallel --ungroup --bar -j 10 python examples/scannetpp_dataset.py +scene_idx={} ::: $(seq 0 $(($num_scenes-1)))
