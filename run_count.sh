#!/bin/bash
#PBS -q short-c
#PBS -l select=1
#PBS -W group_list=go25
#PBS -j oe

cd quasi_identifer
source env/bin/activate
python analyze_dictionary_cooccurrence.py --sample-ratio 1 --min-count-profile 100 --min-odds-ratio 2 --row-batch-size 16 --pool-chunksize 1 --progress-every 10