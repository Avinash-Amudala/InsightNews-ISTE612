# InsightNews: Personalized News Sentiment Analyzer

## Project Setup

1. **Clone the repository**:
    ```sh
    git clone 
    cd InsightNews-ISTE612
    ```

2. **Set up the virtual environment**:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

4. **Set up environment variables**:
   Create a `.env` file in the root directory with your API keys.
    ```env
    NEWS_API_KEY=
    MEDIASTACK_API_KEY=
    ```

## Running the Flask Application

1. **Navigate to the `src` directory**:
    ```sh
    cd src
    ```

2. **Run the application**:
    ```sh
    python app.py
    ```

3. **Open your browser and go to**:
    ```
    http://127.0.0.1:5000
    ```

## Running Sentiment Analysis

1. **Navigate to the `src` directory**:
    ```sh
    cd src
    ```

2. **Run the sentiment analysis script**:
    ```sh
    python sentiment_analysis.py
    ```

## Running Tests

1. **Navigate to the `src` directory**:
    ```sh
    cd src
    ```

2. **Run the test scripts**:
    ```sh
    python -m unittest test_scripts.py
    ```

## Dependencies

- Python 3.9+
- Flask
- pandas
- transformers
- unittest
