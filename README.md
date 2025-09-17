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
- An API key for your chosen LLM provider (OpenAI, Gemini, Together).
- Set up the following environment variables with your API keys:
```bash
export OPENAI_API_KEY="your-openai-api-key"
export GEMINI_API_KEY="your-gemini-api-key"
export TOGETHER_API_KEY="your-together-api-key"
```

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

## 📊 Results

The final output of the analysis is saved in the `result/` directory as two JSON files:

-   `{MODEL_ID}_att_result.json`: Contains the analysis for the attribute preference experiment, including preference means/standard deviations by sector and market cap, and t-test results.
-   `{MODEL_ID}_str_result.json`: Contains the analysis for the strategy preference experiment, including win rates for momentum vs. contrarian strategies and the Chi-squared test result.
