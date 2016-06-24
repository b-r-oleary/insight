from __future__ import division

import numpy as np
import pandas as pd
import re
from collections import defaultdict
from scipy import sparse

import matplotlib.pyplot as plt
import seaborn as sns

# some NLP and machine learning imports:
from sklearn import cross_validation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import roc_curve, precision_recall_curve

# a pretrained sentiment model
from textblob import TextBlob


class ClassificationModel(object):
    
    def __init__(self, x, y,
                 n_features=500, test_fraction=.3, 
                 auto_fit=True, model=None,
                 vectorizer=None,
                 include_interactions=True,
                 include_sentiment=False,
                 name=None,
                 verbose=False,
                 vectorizer_options=None,
                 threshold=None,
                 set_threshold_params={'recall':0.5}):

        self.name = str(name)
        self.verbose = verbose

        if vectorizer_options is None:
        	vectorizer_options = {}
        self.vectorizer_options = vectorizer_options

        if threshold is None:
        	threshold = .5
        self.threshold = threshold
        self.set_threshold_params = set_threshold_params

        # create a label encoder for y
        self.enc = LabelEncoder()
        self.enc.fit(y)
        
        self.n = len(self.enc.classes_)

        # here are some options for additional features beyond single token bag of words
        self.include_interactions = include_interactions
        self.include_sentiment    = include_sentiment

        if self.include_interactions:
            self.polynomial_features = PolynomialFeatures(degree=2, 
                                                          interaction_only=True, 
                                                          include_bias=True)
        else:
            self.polynomial_features = None
        
        # hold onto raw x and raw y,
        # but denote the raw ones with underscore
        self.x = None
        self._x = x
        self.y = self.enc.transform(y)
        self._y = y
        
        self.vocab = None
        
        self.n_features    = n_features
        self.test_fraction = test_fraction
        
        # assume a logistic regression model
        if model is None:
            model = MultinomialNB()
        if vectorizer is None:
            vectorizer = CountVectorizer
            
        self.vecto = vectorizer
        self.model = model
        
        if auto_fit:
            self.fit_and_cross_validate()

    def set_threshold(self, recall=0.5, precision=None):
    	if precision is None:
    		x = np.array(self.prc['recall']) - recall
    	else:
    		x = np.array(self.prc['precision']) - precision
    	self.threshold = self.prc['thresholds'][
    						np.argmin(np.abs(np.mean(x, axis=0)))
    	]
    	return self.threshold

    def transform(self, x):
        x2 = self.vectorizer.transform(x).toarray()
        if self.include_sentiment:
            sentiment = self.get_sentiment(x)
            
            x2 = np.concatenate([x2, 
                                 sentiment
                                 ], axis=1)
        if self.include_interactions:
            x2 = self.polynomial_features.fit_transform(x2)
        
        return sparse.coo_matrix(x2)

    def get_sentiment(self, x=None, transform=True):
        if x is None:
            x = self._x
        sentiment =  np.array(
                        [TextBlob(sentence). polarity for sentence in x]
                     )
        if transform:
            sentiment = np.array([(sentiment > 0) * (+ sentiment),
                                  (sentiment < 0) * (- sentiment)]).T
        else:
            sentiment = np.expand_dims(sentiment, 1)
            
        return sentiment

    def fit_and_cross_validate(self):
        if self.verbose: print('fitting: ' + self.name)
        self.fit()

        self.discrimination, self.ddiscrimination,\
                                  self.properties, self.roc, self.prc = self.cross_validate()
        if self.set_threshold_params is not None:
        	if isinstance(self.set_threshold_params, dict):
        		self.set_threshold(**self.set_threshold_params)
        	else:
        		self.set_threshold()
        	self.set_threshold_params = None
        	self.discrimination, self.ddiscrimination,\
                                  self.properties, self.roc, self.prc = self.cross_validate()

        self.accuracy = self.properties['accuracy']
        self.precision = self.properties['precision']
        self.recall = self.properties['recall']

        self.fit()

    def get_fit_properties(self):
        d  = self.discrimination
        dd = self.ddiscrimination
        accuracy = np.trace(d)/np.sum(d)
    
    def fit(self):

        if self.vocab is None:
            self.vectorizer = self.vecto()
            self.vectorizer.fit(self._x)
            
            self.x  = self.vectorizer.transform(self._x)
            temp = LogisticRegression()
            temp.fit(self.x, self.y)
            
            features = self.vectorizer.get_feature_names()
            self.vocab = [features[i] for i in 
                          reversed(np.array(np.argsort(np.abs(
                                temp.coef_
                            ))[0])
                          )]
            
            self.vectorizer = self.vecto(vocabulary=self.vocab[:self.n_features],
            							 **self.vectorizer_options)
            self.vectorizer.fit(self._x)
            self.x = self.transform(self._x)
            self.fit()
        else:
            if self.x is None:
                self.x = self.transform(self._x)
            self.model.fit(self.x, self.y)
        return
    
    def predict(self, sentences):
        x = self.transform(sentences)
        try:
            if self.threshold == 0.5:
                return self.model.predict(x)
            else:
                return np.array(
                            self.model.predict_proba(x)[:,1] > self.threshold,
                            dtype=int)
        except:
            return self.model.predict(x)
    
    def predict_proba(self, sentences):
        x = self.transform(sentences)
        return self.model.predict_proba(x)
    
    def regression_plot(self):
        inds = np.argsort(self.model.predict_proba(self.x)[:,1])
        plt.plot(np.sort(self.model.predict_proba(self.x)[:,1]))
        plt.plot([self.y[i] for i in inds],'.', alpha=.1)
        plt.ylabel('probability')
        plt.xlabel('trial index')
        plt.title('')
        plt.ylim([-.1, 1.1])
        plt.title(self.name)
        plt.legend(['prediction',
                    'training data'], loc='upper left')
        return
    
    def cross_validate(self, repetitions=50, test_size=.25):
        
        power = []
        accuracy = []
        properties = {'accuracy' : [],
                      'precision': [],
                      'recall'   : []}


        roc = {'fpr':np.linspace(0, 1, 100),
               'tpr':[]}
        prc = {'thresholds':np.linspace(0, 1, 100),
        	   'recall':[],
               'precision':[],
               'combination':[]}

        for i in range(repetitions):

            X_train, X_test, y_train, y_test = \
                     cross_validation.train_test_split(self.x, self.y, test_size=self.test_fraction)#, random_state=np.random.

            self.model.fit(X_train, y_train)
            y_predict = np.array(self.model.predict_proba(X_test)[:,1] > self.threshold, dtype=int)
            y_score   = self.model.predict_proba(X_test)[:,1]
            
            power.append(self.power(y_test, y_predict))
            properties['accuracy'].append(np.sum(y_predict == y_test)/float(len(y_test)))
            properties['precision'].append(np.sum(y_predict * y_test)/float(np.sum(y_predict)))
            properties['recall'].append(np.sum(y_predict * y_test)/float(np.sum(y_test)))

            fpr, tpr, thresholds = roc_curve(y_test, y_score)
            roc['tpr'].append(
                        np.interp(roc['fpr'], fpr, tpr)
                        )
            precision, recall, thresholds = precision_recall_curve(y_test, y_score)
            thresholds = np.array([0] + list(thresholds))

            p_interp = np.interp(prc['thresholds'], thresholds, precision)
            r_interp = np.interp(prc['thresholds'], thresholds, recall)

            prc['precision'].append(p_interp)
            prc['recall'].append(r_interp)
            prc['combination'].append(p_interp * r_interp/(p_interp + r_interp))

        for k, v in properties.items():
            properties[k] = (np.mean(v), np.std(v))

        return np.mean(power, axis=0), np.std(power, axis=0), properties, roc, prc

    
    def power(self, y_actual, y_predict):
        output = np.zeros((self.n, self.n), dtype=int)
        for y0, y1 in zip(y_actual, y_predict):
            output[y0, y1] += 1
        return output

    def roc_curve(self):
        x = self.roc['fpr']
        y = self.roc['tpr']

        m = np.mean(y, axis=0)
        dm= np.std(y, axis=0)

        plt.plot([0,1],[0,1],'--k')
        plt.fill_between(x, m+dm, m-dm, alpha=0.5, linewidth=0, color=sns.color_palette()[0]) 
        plt.plot(x, np.mean(y, axis=0))
        plt.ylim([0,1])
        plt.xlim([0,1])
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        plt.title('ROC curve for ' + self.name)
        return

    def precision_recall_curve(self):
    	x = self.prc['thresholds']
        y = self.prc['precision']
        z = self.prc['recall']
        # w = self.prc['combination']

        m = np.mean(y, axis=0)
        dm= np.std(y, axis=0)
        n = np.mean(z, axis=0)
        dn= np.std(z, axis=0)
        # p = np.mean(w, axis=0)
        # dp= np.std(w, axis=0)

        # plt.plot([0,1],[0,1],'--k')
        plt.fill_between(x, m+dm, m-dm, alpha=0.5, linewidth=0, color=sns.color_palette()[0]) 
        plt.fill_between(x, n+dn, n-dn, alpha=0.5, linewidth=0, color=sns.color_palette()[1]) 
        # plt.fill_between(x, p+dp, p-dp, alpha=0.5, linewidth=0, color=sns.color_palette()[2])
        plt.plot(x, m, label='precision')
        plt.plot(x, n, label='recall')
        # plt.plot(x, p, label='p r / (p + r)')
        plt.ylim([0,1])
        plt.xlim([0,1])
        plt.xlabel('Decision Threshold')
        #plt.ylabel('Precision')
        plt.title('Precision-Recall curve for ' + self.name)
        plt.legend()
    
    def vary_n_features(self):
        pass





class FaultFindr(object):
    """
    This object contains the model for classifying sentences within amazon reviews
    for:
        - whether or not it describes a laptop failure or not
        - what the sentiment is in this sentence
        - what is being described in this sentence (hard drive, motherboard, screen, touchpad, ...)
    """
    
    not_modes  = ['sentence', 'failure', 'sentiment', 
                  'readable', 'review_index', 'sentence_index',
                  'subjectivity', 'polarity']
    sentiments = ['bad', 'neutral', 'good']
    
    def __init__(self, trained,
                 n_features=500, test_fraction=.15, 
                 auto_fit=True, model=None, verbose=False,
                 include_interactions=None,
                 include_sentiment=None,
                 name=None,
                 vectorizer_options=None):
        
        self.train = trained
        self.modes = [c for c in self.train.columns 
                        if c not in self.not_modes]
        
        self.n_features    = n_features
        self.test_fraction = test_fraction
        
        self.fields = ['failure'] + self.modes

        names = self.get_default_names()

        defaults = {'model': {field: MultinomialNB() for field in self.fields},
                    'name':  names,
                    'n_features': {field: 250
                                   if field in ['failure']
                                   else   500
                                   for field in self.fields},
                    'include_interactions': {field: True
                                             if field in ['failure']
                                             else   False
                                             for field in self.fields},
                    'include_sentiment': {field: True
                                          if field in ['failure']
                                          else   False
                                          for field in self.fields},
                    'vectorizer_options': {field:{} for field in self.fields}}

        inputs = {}
        for k, v in defaults.items():
            inputs[k] = self._handle_inputs(locals()[k],
                                              defaults[k])

        kwargs = {field: {k: v[field] for k, v in inputs.items()}
                  for field in self.fields}

        # if model is None:
        #     model = {field: MultinomialNB() for field in fields}
        # elif isinstance(model, dict):
        #     for field in fields:
        #         if field not in model.keys():
        #             model[field] = MultinomialNB()
        # else:
        #     model = {field: model for field in fields}
        
        self.models = {field: ClassificationModel(list(self.train.sentence),
                                              list(self.train[field]),
                                              verbose=verbose,
                                              **kwargs[field])
                       for field in self.fields}
        
        accuracy = {k: model.accuracy for k, model in self.models.items()}
        self.accuracy = pd.DataFrame(accuracy, index=['mean', 'std']).T
        self.discrimination = {k: model.discrimination for k, model in self.models.items()}
        
    @staticmethod
    def read_sql(conn, **kwargs):
        trained = pd.read_sql_table("TrainSentences", conn)
        return FaultFindr(trained, **kwargs)

    def get_default_names(self):
        names = defaultdict(lambda : '', {field: field for field in self.fields})
        names['build_design_quality']     = 'build quality'
        names['cooling_system_fan_noise'] = 'cooling system'
        names['customer_service_returns'] = 'customer service'
        names['disk_drive']               = 'CD/DVD drive'
        names['hard_drive']               = 'hard drive'
        names['freeze_crash_boot_issue']  = 'boot-up and crashes'
        names['motherboard_gpu_memory_processor'] = 'internal components'
        names['operating_system_bios']    = 'operating system'
        names['speed_power_responsive']   = 'speed and responsiveness'
        names['wifi_bluetooth_internet']  = 'wifi and internet'
        return names

    def _handle_inputs(self, arg, default):
        if arg is None:
            arg = {field: (default[field]
                   if isinstance(default, dict)
                   else  default)
                   for field in self.fields}
        elif isinstance(arg, dict):
            for field in self.fields:
                if field not in arg.keys():
                    if isinstance(default, dict):
                        d = default[field]
                    else:
                        d = default
                    arg[field] = d
        else:
            arg = {field:arg for field in self.fields}
        return arg
    
    def predict(self, text, output='list'):
        
        # allow for preparsed sentences as input, or multiple sentences in text:
        if isinstance(text, list):
            sentences = text
        else:
            sentences  = text.split('.')
            
        # perform predictions based on the model:
        prediction = {}
        for name, model in self.models.items():
            prediction[name] = model.predict(sentences)
            
        # allow for multiple output types:
        if output == 'list':
            classifications = []
            for i in range(len(sentences)):
                classifications.append([])
                for k in prediction.keys():
                    if prediction[k][i]:
                        classifications[i].append(k)
            return classifications
        else:
            return pd.DataFrame(prediction)

    def properties(self):
        prop_names = ['accuracy', 'daccuracy', 'precision', 'dprecision',
                 'recall', 'drecall']
        props = {i:[] for i in prop_names}
        for name, model in self.models.items():
            for k, v in model.properties.items():
                props[k].append(v[0])
                props['d' + k].append(v[1])
        return pd.DataFrame(props, 
                            columns=prop_names,
                            index=self.models.keys())

        
    def predict_proba(self, text):
        # allow for preparsed sentences as input, or multiple sentences in text:
        if isinstance(text, list):
            sentences = text
        else:
            sentences  = text.split('.')
        sentences  = [sentence for sentence in sentences 
                            if sentence not in ['', ' ']]
            
        # perform predictions based on the model:
        prediction = {}
        for name, model in self.models.items():
            prediction[name] = model.predict_proba(sentences)[:,1]
            
        # allow for multiple output types:
        return pd.DataFrame(prediction).T

    def __getitem__(self, i):
        return self.models[i]