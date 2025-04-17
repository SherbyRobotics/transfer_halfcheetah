#!/bin/sh

python3 performance_data_extractor.py $1
python3 observation_data_extractor.py $1
python3 action_data_extractor.py $1