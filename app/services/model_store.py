import logging
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from app.core.config import PROJECT_ROOT

model = None
tokenizer = None

MODEL_PATH = PROJECT_ROOT / "DL_Models" / "sentiment_rnn_model.keras"
TOKENIZER_PATH = PROJECT_ROOT / "DL_Models" / "tokenizer.pickle"

logger = logging.getLogger(__name__)

def load_DL_model():
    global model

    if model is None:
        with open(MODEL_PATH, "rb") as f:
            logger.info("loading DL model....")
            logger.info(f"tensorflow version : {tf.__version__}")
            model = load_model(MODEL_PATH)


def load_tokenizer():
    global tokenizer

    if tokenizer is None:
        with open(TOKENIZER_PATH, 'rb') as handle:
            logger.info("loading tokenizer...")
            tokenizer = pickle.load(handle)



def get_model():
    if model is None:
        load_DL_model()
    return model

def get_tokenizer():
    if tokenizer is None:
        load_tokenizer()
    return tokenizer
