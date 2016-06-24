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
    """
    this is an object that wraps a binary sklearn classification model,
    but also includes a bag of words vectorizer, and some
    validatation and plotting methods.

    methods:
    set_threshold:  set the decision threshold for the model
    transform:      given input list of sentences, output feature matrix
    get_sentiment:  given input list of setnences, get TextBlob sentiment
    fit_and_cross_validate: fit the model, evaluate the performance, the refit
    predict:        predict the output to the model
    predict_proba:  predict the probability for the output to the model
    regression_plot: create a plot showing the model discrimination between
                    the two classes
    roc_curve:      create a plot with an ROC curve
    precision_recall_curve: create a plot with a precision-recall curve.
    """
    
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
        """
        inputs:
        x: (list of strings) - x training data
        y: (list or array of integers or booleans) - y training data
        n_features: (int) - number of words to keep in the bag of words model
                            after ranking the features by how predictive they are
                            in an L1 regularized logistic regression model
        test_fraction: (float 0-1) - fraction to use for the test set when doing
                            evaluating the performance of the model
        auto_fit: (boolean) - whether or not to fit the classification model on start-up
        model: (sklearn classification model) - model to use for classification (default to MultinomialNB())
        vectorizer: (sklearn vectorization model) - model to use for bag of words vectorization (default to CountVectorizer)
        include_interactions: (boolean) - whether or not to include interactions terms between words as additional features in the model
        include_sentiment: (boolean) - whether or not to include sentiment as additional features (using the TextBlob pretrained model)
        name: (string) - name of the model (mostly for plotting purposes and future reference)
        verbose: (string) - whether or not to provide details regarding the fitting method
        vectorizer_options: (dictionary) - kwarg options for the vectorizer (for example {'binary':True})
        threshold: (float) - decision threshold - if not set, automatically set decision threshold
        set_threshold_params: (dictionary) - if this is None, use default (.50) decision threshold, otherwise, can set the decision
                                            threshold at a point with fixed recall or precision, for example {'recall': 0.5}
        """

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
        """
        automatically set decision threshold, based on the
        instructions in the input field "set_threshold_params"

        By default, sets the decision threshold to the point
        with fixed recall of 0.5. If precision is input,
        then the decision threshold is set to the point with
        fixed precision.
        """
    	if precision is None:
    		x = np.array(self.prc['recall']) - recall
    	else:
    		x = np.array(self.prc['precision']) - precision
    	self.threshold = self.prc['thresholds'][
    						np.argmin(np.abs(np.mean(x, axis=0)))
    	]
    	return self.threshold

    def transform(self, x):
        """
        transforms an input list of sentences
        to 2d sparse matrix of features according to
        the feature model.

        The feature model can include bag of words
        of varying vocab length, interactions between
        words, and sentiment.
        """
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
        """
        obtain sentiment features from the pretrained
        TextBlob model for an input list of sentences x.

        In order to transform TextBlob.polarity (a float from -1 to 1)
        to a pseudo word count for the Naive Bayes model, I create two 
        new features: one for positive polarity between 0-1, and one
        for negative polarity between 0-1.
        """
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
        """
        1. fit the model
        2. refit the model with a smaller train set and evaluate on a test set
        3. fit the model again with all of the data
        """
        if self.verbose: print('fitting: ' + self.name)
        self.fit()

        self.confusion_matrix, self.dconfusion_matrix,\
                                  self.properties, self.roc, self.prc = self.cross_validate()
        if self.set_threshold_params is not None:
        	if isinstance(self.set_threshold_params, dict):
        		self.set_threshold(**self.set_threshold_params)
        	else:
        		self.set_threshold()
        	self.set_threshold_params = None
        	self.confusion_matrix, self.dconfusion_matrix,\
                                  self.properties, self.roc, self.prc = self.cross_validate()

        self.accuracy = self.properties['accuracy']
        self.precision = self.properties['precision']
        self.recall = self.properties['recall']

        self.fit()

    def get_fit_properties(self):
        d  = self.confusion_matrix
        dd = self.dconfusion_matrix
        accuracy = np.trace(d)/np.sum(d)
    
    def fit(self):
        """
        fit the model:

        if the model has not yet been fit, then first run a linearized linear regression on a simple bag of words model with the full set of features, then truncated the vocabulary based on the most predictive self.n_features features and train a new vectorizer based on this
        new vocabulary.

        if the model has already been fit, then the vectorizer does not have
        to be defined and the model can just be fit right away.
        """

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
        """
        predict the outcomes for each of the input list of sentences
        """
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
        """
        predict the probability of certain outcomes for each of the input
        list of sentences.
        """
        x = self.transform(sentences)
        return self.model.predict_proba(x)
    
    def regression_plot(self):
        """
        Create a regression plot - this consists of plotting the
        predicted probability for the classification model
        vs the training set outcomes ranked in order of
        model probability. This is one way to visualize how
        how well the model separates the training set into the two
        categories.
        """
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
        """
        Using the specified self.test_fraction, evaluate properties of the
        model evaluated on a test set after training the model on a smaller set.
        Obtain confidence intervals on these properties, assuming normal statistics
        and evaluating the mean and standard deviation.
        Also evaluate the ROC curve, and precision/recall curve.
        """

        confusion_matrix = []
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
            
            confusion_matrix.append(self.get_confusion_matrix(y_test, y_predict))
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

        return np.mean(confusion_matrix, axis=0), np.std(confusion_matrix, axis=0), properties, roc, prc

    
    def get_confusion_matrix(self, y_actual, y_predict):
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
        """
        inputs:

        trained: training data dataframe
        n_features: (int) number of features to use for the model
        test_fraction: (float 0-1) fraction of the dataset to use
                    as a hold-out set for evaluating the model
        auto_fit: (boolean) whether or not to fit the model on
                    initialization of the object
        model: (sklearn model) the classification model to use
        verbose: (boolean) whether or not to print status updates
                when training the model
        include_interactions: (boolean) whether or not to take into account
                non-local correlations between words as additional
                features
        include_sentiment: (boolean) whether or not to include sentiment as
                additional features
        name: (string) name for the models
        vectorizer_options: (dictionary) kwarg arguments for the vectorizer
                objects.

        all of these inputs exept verbose and trained can be input as
        dictionaries instead to feed different options to the different
        models, with the model name as the key.
        """
        
        self.train = trained
        self.modes = [c for c in self.train.columns 
                        if c not in self.not_modes]
        
        self.n_features    = n_features
        self.test_fraction = test_fraction
        
        self.fields = ['failure'] + self.modes

        names = self.get_default_names()

        # default options for the inputs to the models:
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

        # reconciling the inputs to this object with the default options:
        inputs = {}
        for k, v in defaults.items():
            inputs[k] = self._handle_inputs(locals()[k],
                                              defaults[k])

        kwargs = {field: {k: v[field] for k, v in inputs.items()}
                  for field in self.fields}

        # creating and training each of the models
        self.models = {field: ClassificationModel(list(self.train.sentence),
                                              list(self.train[field]),
                                              verbose=verbose,
                                              **kwargs[field])
                       for field in self.fields}
        
        # consolidation of the model properties
        accuracy = {k: model.accuracy for k, model in self.models.items()}
        self.accuracy = pd.DataFrame(accuracy, index=['mean', 'std']).T
        self.confusion_matrix = {k: model.confusion_matrix for k, model in self.models.items()}
        
    @staticmethod
    def read_sql(conn, **kwargs):
        """
        obtain training data from an sql database with
        table name "TrainSentences", and then create a
        FaultFindr object.
        """
        trained = pd.read_sql_table("TrainSentences", conn)
        return FaultFindr(trained, **kwargs)

    def get_default_names(self):
        """
        here I am storing a mapping between
        the model keys and the names that I would like
        to attribute to those models
        """
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
        """
        consolidate model inputs with the default values
        """
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
        """
        for an input list of sentences (text), return
        predictions for all of the models.
        If output='list', this method will return a list
        of all of the names of the models for which a given sentence
        produced positive results. If output='df', this method
        will produce a dataframe with booleans indicating
        whether or not the model produced a positive result
        """
        
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
        """
        given an input list of sentences, output 
        a dataframe of the predicted probability for
        each sentence for each model
        """
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
        """
        as a convenience, enable bracket indexing on the
        object to be equivalent to bracket indexing on the
        models attribute.
        """
        return self.models[i]