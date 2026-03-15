#!/bin/bash

# Ensure the task name is passed as an argument
if [ -z "$1" ]; then
    echo "Usage: $0 <task_name> <data_num> <checkpoint_num>"
    exit 1
fi
if [ -z "$2" ]; then
    echo "Usage: $0 <task_name> <data_num> <checkpoint_num>"
    exit 1
fi
if [ -z "$3" ]; then
    echo "Usage: $0 <task_name> <data_num> <checkpoint_num>"
    exit 1
fi

CONFIG_NAME="$1"
DATA_NUM="$2"
# CHECKPOINT_NUM="$3"
DEBUG_MODE="$3"
TASK_NAME="$4"
AGENT_CONFIG="$5"
RESTORE_PATH="$6"
RESTORE_EPOCH="$7"
GOAL_DIR="$8"
OBSERVATION="$9"
# Optional arguments (default to None if not provided)
AGENT_HIGH_ALPHA="${10:-None}"
AGENT_LOW_ALPHA="${11:-None}"
AGENT_ENCODER="${12:-None}"
AGENT_LOW_ACTOR_REP_GRAD="${13:-None}"
AGENT_P_AUG="${14:-None}"
AGENT_SUBGOAL_STEPS="${15:-None}"
AGENT_ALPHA="${16:-None}"
AGENT_NAME="${17:-None}"
# Generate a log file with timestamp
LOG_FILE="eval_results_${TASK_NAME}_${DATA_NUM}_${RESTORE_EPOCH}_${AGENT_NAME}_$(date +"%Y%m%d_%H%M%S").log"

echo "Evaluating task: $TASK_NAME"
echo "Evaluating task: $TASK_NAME"  >> "$LOG_FILE"
TOTAL=0
SUCCESS=0
SUCCESS_RATE=0

# Enable --quiet flag if DEBUG_MODE is "0" or "false"
QUIET_FLAG=""
if [[ "$DEBUG_MODE" == "0" || "$DEBUG_MODE" == "false" ]]; then
    QUIET_FLAG="--quiet"
fi

for SEED in {1000..1099}
do
    echo "Running evaluation with seed $SEED for task $CONFIG_NAME..."
    OUTPUT=""
    if [[ "$DEBUG_MODE" == "0" || "$DEBUG_MODE" == "false" ]]; then
        OUTPUT=$(python ./policy/OGCRL/eval_multi_dp.py \
                --config="$CONFIG_NAME" \
                --data-num=$DATA_NUM \
                --render-mode="sensors" \
                -o="rgb" \
                -b="cpu" \
                -n 1 \
                -s $SEED \
                --agent_config "$AGENT_CONFIG" \
                --restore_path "$RESTORE_PATH" \
                --restore_epoch "$RESTORE_EPOCH" \
                --goal_dir "$GOAL_DIR" \
                --observation "$OBSERVATION" \
                --agent_high_alpha "$AGENT_HIGH_ALPHA" \
                --agent_low_alpha "$AGENT_LOW_ALPHA" \
                --agent_encoder "$AGENT_ENCODER" \
                --agent_low_actor_rep_grad "$AGENT_LOW_ACTOR_REP_GRAD" \
                --agent_p_aug "$AGENT_P_AUG" \
                --agent_subgoal_steps "$AGENT_SUBGOAL_STEPS" \
                --agent_alpha "$AGENT_ALPHA" \
                --task_name "$TASK_NAME" \
                $QUIET_FLAG)
    else
        OUTPUT=$(python ./policy/OGCRL/eval_multi_dp.py \
                --config="$CONFIG_NAME" \
                --data-num=$DATA_NUM \
                --render-mode="sensors" \
                -o="rgb" \
                -b="cpu" \
                -n 1 \
                -s $SEED \
                --agent_config "$AGENT_CONFIG" \
                --restore_path "$RESTORE_PATH" \
                --restore_epoch "$RESTORE_EPOCH" \
                --goal_dir "$GOAL_DIR" \
                --observation "$OBSERVATION" \
                --agent_high_alpha "$AGENT_HIGH_ALPHA" \
                --agent_low_alpha "$AGENT_LOW_ALPHA" \
                --agent_encoder "$AGENT_ENCODER" \
                --agent_low_actor_rep_grad "$AGENT_LOW_ACTOR_REP_GRAD" \
                --agent_p_aug "$AGENT_P_AUG" \
                --agent_subgoal_steps "$AGENT_SUBGOAL_STEPS" \
                --agent_alpha "$AGENT_ALPHA" \
                --task_name "$TASK_NAME" \
                $QUIET_FLAG)
    fi
    echo "$OUTPUT"
    LAST_LINE=$(echo "$OUTPUT" | tail -n 1)  # Get the last line of output
    FINE=0
    # If the output contains "success", the task succeeded
    if [[ $LAST_LINE == *"success"* ]]; then
        FINE=1
        SUCCESS=$((SUCCESS + 1))
    fi
    TOTAL=$((TOTAL + 1))
    SUCCESS_RATE=$(echo "scale=4; $SUCCESS / $TOTAL * 100" | bc)
    echo "$SEED, $FINE, $SUCCESS_RATE%" >> "$LOG_FILE"
    echo "Seed $SEED done. Success Rate: $SUCCESS_RATE%"
done
echo "Total: $TOTAL, Success: $SUCCESS, Success Rate: $SUCCESS_RATE%" >> "$LOG_FILE"

echo "Evaluation completed. Results saved in $LOG_FILE."
