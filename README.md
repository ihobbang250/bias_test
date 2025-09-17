# LLM Financial Bias Test Suite

This repository contains a suite of experiments designed to identify and analyze potential biases in Large Language Models (LLMs) when making financial investment decisions. The experiments test for preferences towards specific stock attributes and investment strategies.

## 🧪 Experiments

This suite includes two main experiments:

### 1. Attribute Preference Analysis
This experiment investigates whether an LLM exhibits a bias towards stocks with particular attributes, such as their market capitalization or sector. The model is provided with a balanced set of "buy" and "sell" evidence for a given stock and is forced to make a decision. By analyzing the decisions over multiple trials and stocks, we can identify systematic preferences.

-   **`preference_attribute.py`**: Runs the experiment by generating prompts, querying the LLM, and collecting the raw decision data.
-   **`result_attribute.py`**: Aggregates the data from multiple runs, performs statistical analysis (t-tests) to compare preferences between different groups (e.g., high-preference vs. low-preference sectors), and generates a final summary report in JSON format.

### 2. Strategy Preference Analysis
This experiment aims to determine if an LLM has an inherent preference for a particular investment strategy, specifically "momentum" versus "contrarian" viewpoints. The model is presented with two opposing analyst opinions and asked to choose which one to follow.

-   **`preference_strategy.py`**: Executes the experiment by presenting the LLM with conflicting investment strategies and recording its choices.
-   **`result_strategy.py`**: Analyzes the choices to calculate the "win rate" for each strategy. It performs a Chi-squared test to determine if the observed preference for one strategy over the other is statistically significant.

## 🚀 Getting Started

### Prerequisites
- Python 3.x
- Pandas
- SciPy
- An API key for your chosen LLM provider (OpenAI, Gemini, or Together).

### Installation
1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```
2.  Install the required Python packages:
    ```bash
    pip install pandas scipy numpy openai google-genai together
    ```

### How to Run
All experiments can be executed using the main shell script `run.sh`.

1.  **Configure the experiment**: Open `run.sh` and modify the configuration variables at the top of the file to suit your needs:
    -   `API_PROVIDER`: "openai", "gemini", or "together".
    -   `MODEL_ID`: The specific model you want to test (e.g., "gpt-4.1-nano").
    -   `TEMPERATURE`: The model's generation temperature.
    -   `OUTPUT_DIR`: Directory to save results.
    -   `MAX_WORKERS`: Number of concurrent API calls.
    -   `NUM_TRIALS`: Number of trials per stock in the attribute test.
    -   `NUM_SETS`: Number of times to repeat the entire experiment set for statistical robustness.

2.  **Execute the script**:
    ```bash
    bash run.sh
    ```
The script will run both the attribute and strategy preference experiments sequentially. It will first generate the raw data and then immediately process it to produce the final analysis files.

## run.sh 주석

```bash
#!/bin/bash

# ==============================================================================
# Script to run LLM bias testing experiments.
# This script executes two main experiments to test for biases in Large Language Models (LLMs)
# in the context of financial investment decisions.
#
# Parameters:
#   API_PROVIDER: The LLM provider to use. Options: "openai", "gemini", "together".
#   MODEL_ID: The specific model ID from the chosen provider.
#   TEMPERATURE: The temperature setting for the LLM's generation. A lower value makes the output more deterministic.
#   OUTPUT_DIR: The directory where result files will be saved.
#   MAX_WORKERS: The number of concurrent threads for making API calls.
#   NUM_TRIALS: The number of trials to run for each stock in the attribute preference experiment.
#   NUM_SETS: The number of independent experiment sets to run.
# ==============================================================================

set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
# Specifies the LLM provider.
API_PROVIDER="openai"
# Specifies the model to be tested.
MODEL_ID="gpt-4.1-nano"
# Sets the creativity of the model's responses.
TEMPERATURE=0.1
# Defines the folder to store experiment results.
OUTPUT_DIR="./result"
# Sets the number of parallel API requests.
MAX_WORKERS=100
# Sets the number of repetitions for each stock test.
NUM_TRIALS=5
# Sets the number of times the entire experiment is repeated.
NUM_SETS=3

# --- Experiment 1: Attribute Preference Test ---
# This experiment tests if the LLM shows a preference for certain stock attributes (e.g., sector, market cap)
# when presented with an equal number of buy and sell arguments.

# Runs the attribute preference experiment.
python preference_attribute.py \
    --api $API_PROVIDER \
    --model-id $MODEL_ID \
    --temperature $TEMPERATURE \
    --output-dir $OUTPUT_DIR \
    --max-workers $MAX_WORKERS \
    --num-trials $NUM_TRIALS \
    --num-sets $NUM_SETS

# Analyzes the results from the attribute preference experiment.
python result_attribute.py \
    --model-id $MODEL_ID \
    --output-dir $OUTPUT_DIR

# --- Experiment 2: Strategy Preference Test ---
# This experiment tests if the LLM prefers a "momentum" or "contrarian" investment strategy.

# Runs the strategy preference experiment.
python preference_strategy.py \
    --api $API_PROVIDER \
    --model-id $MODEL_ID \
    --temperature $TEMPERATURE \
    --output-dir $OUTPUT_DIR \
    --max-workers $MAX_WORKERS \
    --num-sets $NUM_SETS

# Analyzes the results from the strategy preference experiment.
python result_strategy.py \
    --model-id $MODEL_ID \
    --output-dir $OUTPUT_DIR

# --- Completion ---
echo "All experiments and analyses are complete."

```

-----

## README.md

````markdown
# LLM Financial Bias Test Suite

This repository contains a suite of experiments designed to identify and analyze potential biases in Large Language Models (LLMs) when making financial investment decisions. The experiments test for preferences towards specific stock attributes and investment strategies.

## 🧪 Experiments

This suite includes two main experiments:

### 1. Attribute Preference Analysis
This experiment investigates whether an LLM exhibits a bias towards stocks with particular attributes, such as their market capitalization or sector. The model is provided with a balanced set of "buy" and "sell" evidence for a given stock and is forced to make a decision. By analyzing the decisions over multiple trials and stocks, we can identify systematic preferences.

-   **`preference_attribute.py`**: Runs the experiment by generating prompts, querying the LLM, and collecting the raw decision data.
-   **`result_attribute.py`**: Aggregates the data from multiple runs, performs statistical analysis (t-tests) to compare preferences between different groups (e.g., high-preference vs. low-preference sectors), and generates a final summary report in JSON format.

### 2. Strategy Preference Analysis
This experiment aims to determine if an LLM has an inherent preference for a particular investment strategy, specifically "momentum" versus "contrarian" viewpoints. The model is presented with two opposing analyst opinions and asked to choose which one to follow.

-   **`preference_strategy.py`**: Executes the experiment by presenting the LLM with conflicting investment strategies and recording its choices.
-   **`result_strategy.py`**: Analyzes the choices to calculate the "win rate" for each strategy. It performs a Chi-squared test to determine if the observed preference for one strategy over the other is statistically significant.

## 🚀 Getting Started

### Prerequisites
- An API key for your chosen LLM provider (OpenAI, Gemini, or Together).

### Installation
1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```
2.  Install the required Python packages:
    ```bash
    pip install pandas scipy
    ```

### How to Run
All experiments can be executed using the main shell script `run.sh`.

1.  **Configure the experiment**: Open `run.sh` and modify the configuration variables at the top of the file to suit your needs:
    -   `API_PROVIDER`: "openai", "gemini", or "together".
    -   `MODEL_ID`: The specific model you want to test (e.g., "gpt-4.1-nano").
    -   `TEMPERATURE`: The model's generation temperature.
    -   `OUTPUT_DIR`: Directory to save results.
    -   `MAX_WORKERS`: Number of concurrent API calls.
    -   `NUM_TRIALS`: Number of trials per stock in the attribute test.
    -   `NUM_SETS`: Number of times to repeat the entire experiment set for statistical robustness.

2.  **Execute the script**:
    ```bash
    bash run.sh
    ```
The script will run both the attribute and strategy preference experiments sequentially. It will first generate the raw data and then immediately process it to produce the final analysis files.

## 📊 Results

The final output of the analysis is saved in the `result/` directory as two JSON files:

-   `{MODEL_ID}_att_result.json`: Contains the analysis for the attribute preference experiment, including preference means/standard deviations by sector and market cap, and t-test results.
-   `{MODEL_ID}_str_result.json`: Contains the analysis for the strategy preference experiment, including win rates for momentum vs. contrarian strategies and the Chi-squared test result.
```
