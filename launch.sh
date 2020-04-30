#!/bin/bash
# Add this directory to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:`dirname "$(realpath $0)"`
dask-scheduler --local-directory work-dir &

# Launch one worker per GPU
dask-cuda-worker localhost:8786 --nthreads 4 --local-directory work-dir &
python publish_data.py

gunicorn "dash_rapids_mortgage.app:get_server()" --timeout 60 -b :8050 --workers 6
wait
