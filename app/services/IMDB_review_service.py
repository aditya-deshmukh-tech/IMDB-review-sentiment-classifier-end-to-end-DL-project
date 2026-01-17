import re
import nltk
from keras.src.utils import pad_sequences
from nltk.corpus import stopwords
from .model_store import get_model, get_tokenizer
from ..models.model_objs import Review

nltk.download('stopwords')

max_len = 641

def preprocess_text(text):
    text = text.lower()

    text = re.sub(r'<.*?>', '', text)

    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)

    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])

    text = ' '.join(text.split())

    return text

async def classify_IMDB_review_sentiment(review: Review):
    sentence = preprocess_text(review.sentence)

    sequence = get_tokenizer().texts_to_sequences([sentence])
    padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post')

    prediction = get_model().predict(padded_sequence)
    if prediction > 0.5:
        return {"sentiment": "positive"}
    else:
        return {"sentiment": "negative"}
