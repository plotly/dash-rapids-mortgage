#!/bin/bash
# Add this directory to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:`dirname "$(realpath $0)"`
dask-scheduler --local-directory work-dir &
export CUDA_VISIBLE_DEVICES=0
dask-cuda-worker tcp://192.168.1.193:8786 --memory-limit 32GB --local-directory work-dir &
python publish_data.py
python dash_rapids_mortgage/app.py
wait
