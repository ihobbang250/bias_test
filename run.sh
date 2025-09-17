#!/bin/bash

# ==============================================================================
# api_provider: together, openai, gemini
# ==============================================================================

set -e

API_PROVIDER="openai"
MODEL_ID="gpt-4.1-nano"
TEMPERATURE=0.1
OUTPUT_DIR="./test_result"
MAX_WORKERS=100
NUM_TRIALS=5
NUM_SETS=3


python preference_vol.py \
    --api $API_PROVIDER \
    --model-id $MODEL_ID \
    --temperature $TEMPERATURE \
    --output-dir $OUTPUT_DIR \
    --max-workers $MAX_WORKERS \
    --num-trials $NUM_TRIALS \
    --num-sets $NUM_SETS 

python result_vol.py --model-id $MODEL_ID --output-dir $OUTPUT_DIR


python preference_int.py \
    --api $API_PROVIDER \
    --model-id $MODEL_ID \
    --temperature $TEMPERATURE \
    --output-dir $OUTPUT_DIR \
    --max-workers $MAX_WORKERS \
    --num-sets $NUM_SETS 

python result_int.py --model-id $MODEL_ID --output-dir $OUTPUT_DIR
