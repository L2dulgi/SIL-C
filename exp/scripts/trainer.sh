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
DEC=""
ACTION_CHUNK=""

# Evaluation noise parameters
EVAL_NOISE=""
EVAL_NOISE_TYPE=""
EVAL_NOISE_SCALE=""
EVAL_NOISE_CLIP=""
EVAL_NOISE_SEED=""

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
    --dec)
      DEC="$2"
      shift 2
      ;;
    --action_chunk)
      ACTION_CHUNK="$2"
      shift 2
      ;;
    --expid)
        EXPID="$2"
        shift 2
        ;;
    --eval_noise)
        EVAL_NOISE="--eval_noise"
        shift 1
        ;;
    --eval_noise_type)
        EVAL_NOISE_TYPE="$2"
        shift 2
        ;;
    --eval_noise_scale)
        EVAL_NOISE_SCALE="$2"
        shift 2
        ;;
    --eval_noise_clip)
        EVAL_NOISE_CLIP="$2"
        shift 2
        ;;
    --eval_noise_seed)
        EVAL_NOISE_SEED="$2"
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
echo "EXPID=$EXPID | DEC=$DEC | ACTION_CHUNK=$ACTION_CHUNK"
if [ -n "$EVAL_NOISE" ]; then
  echo "EVAL_NOISE: enabled | TYPE=$EVAL_NOISE_TYPE | SCALE=$EVAL_NOISE_SCALE | CLIP=$EVAL_NOISE_CLIP | SEED=$EVAL_NOISE_SEED"
fi

# Run experiments with seeds from START_SEED to (START_SEED + NUM_EXPS - 1)
end_seed=$((START_SEED + NUM_EXPS - 1))
for seed in $(seq $START_SEED $end_seed); do
  while [ $(jobs -r | wc -l) -ge $max_jobs ]; do
    sleep 1
  done

  echo "Starting process with seed = $seed"
  
  # Build command with optional noise parameters
  cmd="python exp/trainer.py \
    --exp_id $EXPID \
    --seed $seed \
    --env $ENV \
    --scenario_type $SCENARIO_TYPE \
    --sync_type $SYNC_TYPE \
    --algorithm $ALGORITHM \
    --lifelong $LIFELONG \
    --dist_type $dist_type \
    --do_eval"

  # Add decoder parameter if specified
  [ -n "$DEC" ] && cmd="$cmd --dec $DEC"

  # Add action_chunk parameter if specified
  [ -n "$ACTION_CHUNK" ] && cmd="$cmd --action_chunk $ACTION_CHUNK"

  # Add noise parameters if enabled
  if [ -n "$EVAL_NOISE" ]; then
    cmd="$cmd $EVAL_NOISE"
    cmd="$cmd --eval_noise_type $EVAL_NOISE_TYPE"
    cmd="$cmd --eval_noise_scale $EVAL_NOISE_SCALE"
    [ -n "$EVAL_NOISE_CLIP" ] && cmd="$cmd --eval_noise_clip $EVAL_NOISE_CLIP"
    [ -n "$EVAL_NOISE_SEED" ] && cmd="$cmd --eval_noise_seed $EVAL_NOISE_SEED"
  fi
  
  # Execute command in background
  eval "$cmd &"
done

wait
