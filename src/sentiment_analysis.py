from transformers import pipeline

def setup_sentiment_analyzer():
    return pipeline('sentiment-analysis', model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

def analyze_sentiment(analyzer, text):
    return analyzer(text)[0]['label']

def setup_summarizer():
    return pipeline('summarization', model="sshleifer/distilbart-cnn-12-6")

def summarize_content(summarizer, text, max_length=130, min_length=30):
    input_length = len(text.split())
    max_len = min(max_length, input_length)
    min_len = min(min_length, input_length // 2)
    return summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)[0]['summary_text']
