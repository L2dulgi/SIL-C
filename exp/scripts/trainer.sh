#!/bin/bash

# Handle termination signals (SIGINT, SIGTERM) to kill all child processes.
trap "echo 'Termination signal received - killing all child processes...'; kill 0" SIGINT SIGTERM

# Default values
export CUDA_VISIBLE_DEVICES=0
ENV="kitchen"
SCENARIO_TYPE="objective"
SYNC_TYPE="sync"
ALGORITHM="ptgm"
LIFELONG="ft"
START_SEED=0
EXPID=""
NUM_EXPS=5
max_jobs=2
dist_type="maha"

# Parse long-form arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      export CUDA_VISIBLE_DEVICES="$2"
      shift 2
      ;;
    --env)
      ENV="$2"
      shift 2
      ;;
    --sc)
      SCENARIO_TYPE="$2"
      shift 2
      ;;
    --sy)
      SYNC_TYPE="$2"
      shift 2
      ;;
    --al)
      ALGORITHM="$2"
      shift 2
      ;;
    --ll)
      LIFELONG="$2"
      shift 2
      ;;
    --j)
      max_jobs="$2"
      shift 2
      ;;
    --start_seed)
      START_SEED="$2"
      shift 2
      ;;
    --num_exps)
      NUM_EXPS="$2"
      shift 2
      ;;
    --dist_type)
      dist_type="$2"
      shift 2
      ;;
    --expid)
        EXPID="$2"
        shift 2
        ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "Running with:"
echo "ENV=$ENV | SCENARIO_TYPE=$SCENARIO_TYPE | SYNC_TYPE=$SYNC_TYPE | ALGORITHM=$ALGORITHM | LIFELONG=$LIFELONG"
echo "START_SEED=$START_SEED | NUM_EXPS=$NUM_EXPS | max_jobs=$max_jobs | CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "EXPID=$EXPID"

# Run experiments with seeds from START_SEED to (START_SEED + NUM_EXPS - 1)
end_seed=$((START_SEED + NUM_EXPS - 1))
for seed in $(seq $START_SEED $end_seed); do
  while [ $(jobs -r | wc -l) -ge $max_jobs ]; do
    sleep 1
  done

  echo "Starting process with seed = $seed"
  python exp/trainer.py \
    --exp_id $EXPID \
    --seed $seed \
    --env $ENV \
    --scenario_type $SCENARIO_TYPE \
    --sync_type $SYNC_TYPE \
    --algorithm $ALGORITHM \
    --lifelong $LIFELONG \
    --dist_type $dist_type \
    --do_eval &
done

wait
