from __future__ import division

import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns

# some NLP and machine learning imports:
from sklearn import cross_validation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# user interface for manual training
from ipywidgets import interact, interactive, fixed
import ipywidgets as widgets
from IPython.display import clear_output, display, HTML


class FailureModel(object):
    
    not_modes = ['sentence', 'failure', 'sentiment', 'readable', 
    			 'review_index', 'sentence_index', 'polarity', 'subjectivity']
    
    def __init__(self, trained, untrained,
                 n_features=500, test_fraction=.15, 
                 auto_fit=True, model=None):
                 #vectorizer=None, vocabulary=None):
        
        self.train = trained
        self.untrain = untrained
        self.modes = [c for c in self.train.columns 
                        if c not in self.not_modes]
        
        self.vocab = None
        
        self.sentiments = list(self.train['sentiment'].unique())
        
        self.n_features    = n_features
        self.test_fraction = test_fraction
        
        # assume a logistic regression model
        if model is None:
            model = LogisticRegression()
        self.failure_model = model
        
        if auto_fit:
            self.fit()
            self.accuracy = self.cross_validate()
            self.fit()
        
    @staticmethod
    def read_sql(conn, **kwargs):
        trained = pd.read_sql_table("TrainSentences", conn)
        untrained = pd.read_sql_table("UnTrainedSentences", conn)
        return FailureModel(trained, untrained, **kwargs)
    
    def to_sql(self, conn):
        self.train.to_sql("TrainSentences", conn, index=False, if_exists='replace')
        self.untrain.to_sql("UnTrainedSentences", conn, index=False, if_exists='replace')
    
    def fit(self):
        """
        fit the logistic regression model to the data
        """
        if self.vocab is None:
            self.vectorizer = CountVectorizer()
            self.vectorizer.fit(list(self.train.sentence))
            
            x  = self.vectorizer.transform(list(self.train.sentence))
            y  = np.array(self.train.failure, dtype=int)
            temp = LogisticRegression()
            temp.fit(x, y)
            
            features = self.vectorizer.get_feature_names()
            self.vocab = [features[i] for i in 
                          reversed(np.array(np.argsort(np.abs(
                                temp.coef_
                            ))[0])
                          )]
            
            self.vectorizer = CountVectorizer(vocabulary=self.vocab[:self.n_features])
            self.vectorizer.fit(list(self.train.sentence))
            self.fit()
        else:
            x  = self.vectorizer.transform(list(self.train.sentence))
            y  = np.array(self.train.failure, dtype=int)
            self.failure_model.fit(x, y)
        return
    
    def regression_plot(self):
        x  = self.vectorizer.transform(list(self.train.sentence))
        y  = np.array(self.train.failure, dtype=int)
        inds = np.argsort(self.failure_model.predict_proba(x)[:,1])
        plt.plot(np.sort(self.failure_model.predict_proba(x)[:,1]))
        plt.plot([y[i] for i in inds],'.', alpha=.1)
        plt.ylabel('probability')
        plt.xlabel('trial index')
        plt.title('')
        plt.ylim([-.1, 1.1])
        plt.title('failure model logistic regression predictions')
        plt.legend(['prediction',
                    'training data'], loc='upper left')
        return
    
    def cross_validate(self, repetitions=50, test_size=.25):
        outputs = {
         'fault_accuracy_train':[],
         'fault_accuracy_test':[],
         'overall_accuracy_train':[],
         'overall_accuracy_test':[]}

        X  = self.vectorizer.transform(list(self.train.sentence))
        y  = np.array(self.train.failure, dtype=int)

        for i in range(repetitions):

            X_train, X_test, y_train, y_test = \
                     cross_validation.train_test_split(X, y, test_size=test_size)#, random_state=np.random.

            self.failure_model.fit(X_train, y_train)
            outputs['fault_accuracy_train'].append(np.sum(self.failure_model.predict(X_train) * y_train)/float(np.sum(y_train)))
            outputs['overall_accuracy_train'].append(np.sum(self.failure_model.predict(X_train) == y_train)/float(len(y_train)))
            outputs['fault_accuracy_test'].append(np.sum(self.failure_model.predict(X_test) * y_test)/float(np.sum(y_test)))
            outputs['overall_accuracy_test'].append(np.sum(self.failure_model.predict(X_test) == y_test)/float(len(y_test)))

        return {k:(np.mean(v), np.std(v)) for k, v in outputs.items()}
    
    def training(self, train_on_true=True):
        
        if train_on_true:
            p = self.failure_model.predict(self.vectorizer.transform(list(self.untrain.sentence)))
            inds = [i for i in self.untrain.index
                    if p[i] == True]
            inds = [inds[i] for i in np.random.permutation(len(inds))]
        else:
            inds = [i for i in np.random.permutation(len(self.untrain))]

        width = 30

        checkboxes = [widgets.Checkbox(description = topic, value=False, width=width)
                      for topic in self.modes]
        cb_container = widgets.VBox(children=checkboxes)
        sentiment_rb = widgets.RadioButtons(description='sentiment',
                                            options=self.sentiments)
        segment = widgets.Textarea(description='review', height=200)
        failure = widgets.RadioButtons(description='failure mode?',
                                        options=['no', 'yes'])
        button = widgets.Button(description='submit')
        right = widgets.VBox(children=[failure, sentiment_rb, segment, button])
        container = widgets.HBox(children=[cb_container, right])

        current = self.untrain.ix[inds[0]]
        segment.value = current.readable
        
        display(container)

        def on_submit(inputs):
            current = self.untrain.ix[inds[0]]
            current.failure = (failure.value == 'yes')
            for c in checkboxes:
                current[c.description] = c.value
            current.sentiment = sentiment_rb.value
            
            sentiment_rb.value = 'neutral'
            for c in checkboxes:
                c.value = False
            failure.value='no'
            
            self.train = self.train.append(current, ignore_index=True)
            self.untrain.drop(inds[0], inplace=True)
            inds.pop(0)
            
            current = self.untrain.ix[inds[0]]
            segment.value = current.readable
            return

        button.on_click(on_submit)