#%%
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

import re
import json
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nlp = spacy.load("en_core_web_sm")

#%%
conv_ids = []
utterances = []
emotions = []
speakers = []
with open('train_dataset.json') as f:
    data = json.load(f)
    for conv_id in data['conversation']:
        for utterance in data['conversation'][conv_id]:
            conv_ids.append(conv_id)
            utterances.append(utterance['text'])
            speakers.append(utterance['speaker'])
            emotions.append(utterance['emotion'])

train_df = pd.DataFrame({'conv_id': conv_ids, 'utterance': utterances, 'speaker': speakers, 'emotion': emotions})
train_df.dropna(inplace=True)
train_df.head()

#%%
conv_ids = []
utterances = []
emotions = []
speakers = []
with open('test_dataset.json') as f:
    data = json.load(f)
    for conv_id in data['conversation']:
        for utterance in data['conversation'][conv_id]:
            conv_ids.append(conv_id)
            utterances.append(utterance['text'])
            speakers.append(utterance['speaker'])
            emotions.append(utterance['emotion'])

test_df = pd.DataFrame({'conv_id': conv_ids, 'utterance': utterances, 'speaker': speakers, 'emotion': emotions})

#%%
emotion_map = {'anger': 0, 'disgust': 1, 'fear': 2, 'joy': 3, 'sadness': 4, 'surprise': 5, 'neutral': 6}
train_df['emotion_label'] = train_df['emotion'].map(emotion_map)
test_df['emotion_label'] = test_df['emotion'].map(emotion_map)

#%%
def preprocess_text(text):
    #text = text.lower()
    #text = re.sub('[^\w\s]', '', text)
    #tokens = word_tokenize(text)
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in text.split()]
    #return ' '.join(stemmed_tokens)
    return stemmed_tokens

#train_df['processed_text'] = train_df['utterance'].apply(preprocess_text)
#test_df['processed_text'] = test_df['utterance'].apply(preprocess_text)

#%%
tfidf_vectorizer = TfidfVectorizer(lowercase=False,tokenizer=preprocess_text, token_pattern=None)
train_tfidf  = tfidf_vectorizer.fit_transform(train_df['utterance'])
test_tfidf  = tfidf_vectorizer.transform(test_df['utterance'])

#%%
grid = {
    'kernel': ['rbf', 'linear'],
    'C': [1,3.5,10]
}
svc_cv = GridSearchCV(estimator=SVC(), param_grid=grid, cv=5)
svc_cv.fit(train_tfidf, train_df["emotion_label"])
print(svc_cv.best_params_)
print(metrics.classification_report(test_df["emotion_label"], svc_cv.predict(test_tfidf)))

#%%
model = SVC()
model.fit(train_tfidf, train_df['emotion_label'])
print(metrics.classification_report(test_df["emotion_label"], model.predict(test_tfidf)))

#%%
count_vectorizer = CountVectorizer(lowercase=False,tokenizer=preprocess_text, token_pattern=None)
train_bow = count_vectorizer.fit_transform(train_df['utterance'])
test_bow = count_vectorizer.transform(test_df['utterance'])

#%%
grid = {
    'kernel': ['rbf', 'linear'],
    'C': [1,3.5,10]
}
svc_cv = GridSearchCV(estimator=SVC(), param_grid=grid, cv=5)
svc_cv.fit(train_bow, train_df["emotion_label"])
print(svc_cv.best_params_)
print(metrics.classification_report(test_df["emotion_label"], svc_cv.predict(test_bow)))

#%%
model = SVC(C=3.5)
model.fit(train_bow, train_df['emotion_label'])
print(metrics.classification_report(test_df["emotion_label"], model.predict(test_bow)))

#%%
model = MultinomialNB()
model.fit(train_tfidf, train_df['emotion_label'])
print(metrics.classification_report(test_df["emotion_label"], model.predict(test_tfidf)))

#%%
rf_grid = { 
    'n_estimators': [50, 100, 150, 200],
    'max_depth' : [2, 4, 6, 8, None],
}
rf_cv = GridSearchCV(estimator=RandomForestClassifier(), param_grid=rf_grid, cv=5)
rf_cv.fit(train_tfidf, train_df["emotion_label"])
print(rf_cv.best_params_)
print(metrics.classification_report(test_df["emotion_label"], rf_cv.predict(test_tfidf)))
