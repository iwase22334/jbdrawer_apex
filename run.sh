#!/bin/bash

mkdir -p output

N_ACTORS=$1
CFG=./apex/config.yaml
python3 ./apex/replay_memory.py --config_file "$CFG" >> output/replay_memory.log 2>&1 &
python3 ./apex/learner.py --config_file "$CFG" --n_actors "$N_ACTORS" --replay_ip 127.0.0.1 >> output/learner.log 2>&1 &

for ((id=0; id<N_ACTORS; id++)); do
    python3 ./apex/actor.py --config_file "$CFG"  --id $id --n_actors "$N_ACTORS" --replay_ip 127.0.0.1 --learner_ip 127.0.0.1 >> output/actor_$id.log 2>&1 &
done

python3 ./apex/evaluate.py --config_file "$CFG" --replay_ip 127.0.0.1  --learner_ip 127.0.0.1 >> output/evaluate.log 2>&1 &
