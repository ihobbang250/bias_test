#!/bin/bash

# ==============================================================================
# Script to run LLM bias testing experiments.
#
# Parameters:
#   API_PROVIDER: The LLM provider to use. Options: "openai", "gemini", "together".
#   MODEL_ID: The specific model ID from the chosen provider.
#   TEMPERATURE: The temperature setting for the LLM's generation.
#   OUTPUT_DIR: The directory where result files will be saved.
#   MAX_WORKERS: The number of concurrent threads for API calls.
#   NUM_TRIALS: The number of trials to run for each stock in the volume experiment.
#   NUM_SETS: The number of experiment sets to run. Each set runs independently.
# ==============================================================================

set -e

# --- Configuration ---
API_PROVIDER="openai"
MODEL_ID="gpt-4.1-nano"
TEMPERATURE=0.1
OUTPUT_DIR="./result"
MAX_WORKERS=100
NUM_TRIALS=5
NUM_SETS=3

python preference_attribute.py \
    --api $API_PROVIDER \
    --model-id $MODEL_ID \
    --temperature $TEMPERATURE \
    --output-dir $OUTPUT_DIR \
    --max-workers $MAX_WORKERS \
    --num-trials $NUM_TRIALS \
    --num-sets $NUM_SETS


python result_attribute.py \
    --model-id $MODEL_ID \
    --output-dir $OUTPUT_DIR

python preference_strategy.py \
    --api $API_PROVIDER \
    --model-id $MODEL_ID \
    --temperature $TEMPERATURE \
    --output-dir $OUTPUT_DIR \
    --max-workers $MAX_WORKERS \
    --num-sets $NUM_SETS

python result_strategy.py \
    --model-id $MODEL_ID \
    --output-dir $OUTPUT_DIR

echo "All experiments and analyses are complete."
