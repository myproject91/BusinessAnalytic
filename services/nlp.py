import re
import nltk
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

nltk.download('stopwords', quiet=True)
vader = SentimentIntensityAnalyzer()

ASPECT_KEYWORDS = {
    'price'      : ['price', 'expensive', 'cheap', 'worth', 'value', 'cost', 'budget'],
    'performance': ['performance', 'fast', 'slow', 'lag', 'speed', 'processor', 'ram'],
    'battery'    : ['battery', 'charge', 'charging', 'backup', 'drain'],
    'display'    : ['display', 'screen', 'resolution', 'brightness', 'visual'],
    'build'      : ['build', 'quality', 'design', 'material', 'keyboard', 'trackpad'],
    'delivery'   : ['delivery', 'shipping', 'packaging', 'packed', 'arrived'],
}


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def classify_sentiment(text: str) -> dict:
    cleaned = clean_text(text)
    if not cleaned:
        return {'label': 'neutral', 'compound': 0.0, 'pos': 0.0, 'neu': 0.0, 'neg': 0.0}

    scores = vader.polarity_scores(cleaned)
    compound = scores['compound']
    label = 'positive' if compound >= 0.05 else 'negative' if compound <= -0.05 else 'neutral'

    return {
        'label'   : label,
        'compound': round(compound, 4),
        'pos'     : round(scores['pos'], 4),
        'neu'     : round(scores['neu'], 4),
        'neg'     : round(scores['neg'], 4)
    }


def run_sentiment_analysis(df: pd.DataFrame, text_col: str) -> dict:
    results = []
    for idx, row in df.iterrows():
        sentiment = classify_sentiment(row[text_col])
        aspects = {}
        cleaned = clean_text(str(row[text_col]))
        for aspect, keywords in ASPECT_KEYWORDS.items():
            for kw in keywords:
                if kw in cleaned:
                    aspects[aspect] = sentiment['label']
                    break

        results.append({
            'index'   : idx,
            'label'   : sentiment['label'],
            'compound': sentiment['compound'],
            'pos'     : sentiment['pos'],
            'neu'     : sentiment['neu'],
            'neg'     : sentiment['neg'],
            'aspects' : aspects
        })

    result_df = pd.DataFrame(results)
    distribution = result_df['label'].value_counts().to_dict()

    aspect_summary = {}
    for row in results:
        for aspect, label in row['aspects'].items():
            if aspect not in aspect_summary:
                aspect_summary[aspect] = {'positive': 0, 'neutral': 0, 'negative': 0}
            aspect_summary[aspect][label] += 1

    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    freq = {}
    for _, row in df.iterrows():
        for word in clean_text(str(row[text_col])).split():
            if word not in stop_words and len(word) > 2:
                freq[word] = freq.get(word, 0) + 1

    top_keywords = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:20]

    return {
        'result_df'     : result_df,
        'distribution'  : distribution,
        'aspect_summary': aspect_summary,
        'top_keywords'  : top_keywords
    }
