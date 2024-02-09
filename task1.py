#%%
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier

import re
import json
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nlp = spacy.load("en_core_web_sm")

#%%
emotion_utterance_texts = []
cause_texts = []
emotion_cause_label = []

with open('train_dataset.json') as f:
    data = json.load(f)
    for conv_id in data['conversation']:
        ec_id_pairs = []
        for emotion_cause_pair in data['emotion-cause_pairs'][conv_id]:
            emotion_utterance_id, emotion = emotion_cause_pair[0].split('_')
            cause_id, cause_text = emotion_cause_pair[1].split('_')
            ec_id_pairs.append((int(emotion_utterance_id), int(cause_id)))

        for i in range(len(data['conversation'][conv_id])):
            for j in range(i+1):
                emotion_utterance_texts.append(data['conversation'][conv_id][i]['text'])
                cause_texts.append(data['conversation'][conv_id][j]['text'])
                
                emotion_cause_label.append(0)
                for ec_id_pair in ec_id_pairs:
                    if data['conversation'][conv_id][i]['utterance_ID'] == ec_id_pair[0] and data['conversation'][conv_id][i]['utterance_ID'] == ec_id_pair[1]:
                        emotion_cause_label[-1] = 1
                        break

train_df = pd.DataFrame({'EmotionUttText': emotion_utterance_texts, 'CauseText': cause_texts, 'ECPair': emotion_cause_label})
train_df.dropna(inplace=True)
train_df.head()

#%%
emotion_utterance_texts = []
cause_texts = []
emotion_cause_label = []

with open('test_dataset.json') as f:
    data = json.load(f)
    for conv_id in data['conversation']:
        ec_id_pairs = []
        for emotion_cause_pair in data['emotion-cause_pairs'][conv_id]:
            emotion_utterance_id, emotion = emotion_cause_pair[0].split('_')
            cause_id, cause_text = emotion_cause_pair[1].split('_')
            ec_id_pairs.append((int(emotion_utterance_id), int(cause_id)))

        for i in range(len(data['conversation'][conv_id])):
            for j in range(i+1):
                emotion_utterance_texts.append(data['conversation'][conv_id][i]['text'])
                cause_texts.append(data['conversation'][conv_id][j]['text'])
                
                emotion_cause_label.append(0)
                for ec_id_pair in ec_id_pairs:
                    if data['conversation'][conv_id][i]['utterance_ID'] == ec_id_pair[0] and data['conversation'][conv_id][i]['utterance_ID'] == ec_id_pair[1]:
                        emotion_cause_label[-1] = 1
                        break

test_df = pd.DataFrame({'EmotionUttText': emotion_utterance_texts, 'CauseText': cause_texts, 'ECPair': emotion_cause_label})
test_df.dropna(inplace=True)
test_df.head()

#%%
def preprocess_text(text):
    #text = text.lower()
    #text = re.sub('[^\w\s]', '', text)
    #tokens = word_tokenize(text)
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in text.split()]
    return stemmed_tokens

tfidf_vectorizer = TfidfVectorizer(lowercase=False,tokenizer=preprocess_text, token_pattern=None)
count_vectorizer = CountVectorizer(lowercase=False,tokenizer=preprocess_text, token_pattern=None)

#%%
class ColumnExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.column]

emotion_utterance_features = Pipeline([
    ("column",ColumnExtractor(column='EmotionUttText')),
    ("fe", tfidf_vectorizer)
])
cause_features = Pipeline([
    ("column",ColumnExtractor(column='CauseText')),
    ("fe", tfidf_vectorizer)
])

fu = FeatureUnion([
    ("ecf",emotion_utterance_features),
    ("cf",cause_features)
])
model = Pipeline([
    ('features', fu),
    ('classifier', RandomForestClassifier())
])

#%%
model.fit(train_df, train_df['ECPair'])
print(metrics.classification_report(test_df["ECPair"], model.predict(test_df)))
