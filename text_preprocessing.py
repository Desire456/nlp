import re
import string
import spacy
import nltk

sp = spacy.load('en_core_web_sm')

all_stopwords = sp.Defaults.stop_words


def preprocess(input_text):
    input_text = input_text.strip()  # remove leading and ending spaces
    input_text = input_text.lower()  # convert to lower case
    input_text = re.sub(r'\d+', '', input_text)  # remove numbers
    input_text = input_text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    input_text = remove_stop_words(input_text) # remove stop words
    input_text = stem(input_text)  # stem words
    input_text = lemmatize(input_text)  # lemmatize words
    return input_text


def remove_stop_words(input_text):
    text_tokens = word_tokenize(input_text)
    tokens_without_sw = [word for word in text_tokens if not word in all_stopwords]
    return ' '.join(tokens_without_sw)


def word_tokenize(input_text):
    doc = sp(input_text)
    return [token.text for token in doc]


def stem(input_text):
    stemmer = nltk.PorterStemmer()
    text_tokens = word_tokenize(input_text)
    text_tokens = [stemmer.stem(word) for word in text_tokens]
    return ' '.join(text_tokens)


def lemmatize(input_text):
    lemmatizer = nltk.WordNetLemmatizer()
    text_tokens = word_tokenize(input_text)
    text_tokens = [lemmatizer.lemmatize(word) for word in text_tokens]
    return ' '.join(text_tokens)


def pos_tag(input_text):  # POS-tagging
    text_tokens = word_tokenize(input_text)
    return nltk.pos_tag(text_tokens)
