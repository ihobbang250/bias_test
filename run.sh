#!/bin/bash

# ==============================================================================
# preference_elicit_volume.py 실행 스크립트
#
# 이 스크립트는 다양한 LLM API를 사용하여 선호도 도출 실험을 실행합니다.
# 스크립트를 실행하기 전에 필요한 API 키를 환경 변수로 설정해야 합니다.
#
# 사용법:
# 1. 스크립트에 실행 권한 부여: chmod +x run_experiments.sh
# 2. 환경 변수 설정:
#    export OPENAI_API_KEY="YOUR_OPENAI_KEY"
#    export GEMINI_API_KEY="YOUR_GEMINI_KEY"
#    export TOGETHER_API_KEY="YOUR_TOGETHER_KEY"
# 3. 스크립트 실행: ./run_experiments.sh
# ==============================================================================

set -e

OUTPUT_DIR="./test_result"
TEMPERATURE=0.1
MAX_WORKERS=100
NUM_TRIALS=5
NUM_SETS=3
MODEL_ID="Qwen/Qwen3-235B-A22B-Instruct-2507-tput"


# python preference_elicit_volume_test.py \
#     --api together \
#     --model-id $MODEL_ID \
#     --temperature $TEMPERATURE \
#     --output-dir $OUTPUT_DIR \
#     --max-workers $MAX_WORKERS \
#     --num-trials $NUM_TRIALS \
#     --num-sets $NUM_SETS 

# python preference_agg_test.py --model-id $MODEL_ID


# python preference_elicit_intensity.py \
#     --api together \
#     --model-id $MODEL_ID \
#     --temperature $TEMPERATURE \
#     --output-dir $OUTPUT_DIR \
#     --max-workers $MAX_WORKERS \
#     --num-sets $NUM_SETS 

python preference_int_agg.py --model-id $MODEL_ID --output-dir $OUTPUT_DIR
