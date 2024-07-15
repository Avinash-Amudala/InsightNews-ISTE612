import unittest
import pandas as pd
from sentiment_analysis import analyze_sentiment, setup_sentiment_analyzer

class TestSentimentAnalysis(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            'cleaned_content': [
                'This is a great product!',
                'I am not happy with the service.',
                'The food was amazing and the staff was friendly.',
                'It was a terrible experience, I will not come back.',
                'The movie was okay, not too bad.',
                'An average experience overall.',
                'Absolutely fantastic service!',
                'Would not recommend this to anyone.',
                'The product exceeded my expectations.',
                'A decent, but not exceptional, offering.'
            ]
        })
        self.analyzer = setup_sentiment_analyzer()

    def test_analyze_sentiment(self):
        result_df = analyze_sentiment(self.df, self.analyzer)
        self.assertIn('sentiment', result_df.columns)
        self.assertEqual(len(result_df), len(self.df))
        self.assertTrue(all(result_df['sentiment'].isin(['POSITIVE', 'NEGATIVE', 'NEUTRAL'])))

if __name__ == '__main__':
    unittest.main()
