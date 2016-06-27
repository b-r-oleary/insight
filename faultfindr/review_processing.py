from __future__ import division

import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import re
from collections import defaultdict
from scipy.stats import poisson
from nltk.tokenize import MWETokenizer
from functools import partial
from nltk.corpus import stopwords

import matplotlib.pyplot as plt
import seaborn as sns


def get_n_grams(text, n_min=1, n_max=3, stop='.'):
    """
    obtain frequency distribution for tokenized text input *text*
    for n_grams of length from *n_min* (int) to *n_max* (int).
    
    if text includes the character *stop*, then n_grams that include
    *stop* will be exluded since it is inferred that those are phrases
    that.
    
    returns a dictionary with integer keys corresponding to the number
    of words in the n-gram (n). the values are dictionaries that have
    the n_grams as keys, and the frequencies are the values.
    
    example:
    {1,{('I',):101, ('will',):20, ...},
     2,{('I','will',):11},
     ...}
    """
    n_grams = {}
    for i in range(n_min, n_max + 1):
        vocab = defaultdict(lambda *args: 0)
        for j in range(len(text) - i + 1):
            key = tuple(text[j:j+i])
            if not(stop in key):
                vocab[key] += 1
        n_grams[i] = vocab
    return n_grams

def combined_n_grams(tokens, n_min=1, n_max=3, stop='.',
                     threshold=5, use_threshold=True):
    """
    calls get_n_grams to obtain n_grams, combines all n_gram frequency dictionaries
    into a single frequency dictionary, and throws out those n_grams with frequencies
    below the *threshold* if *use_threshold* is True.
    """
    n_grams = get_n_grams(tokens,
                          n_min=n_min, n_max=n_max,
                          stop=stop)
    combined = {}
    for i, n_gram in n_grams.items():
        for k,v in n_gram.items():
            if (((i == 1) or (v > threshold))
                or not(use_threshold)):
                combined[k] = v
    return combined

def get_vocab(text):
    """
    obtains the single word frequencies using get_n_gramsw?
    """
    vocab = get_n_grams(text, n_min=1, n_max=1)[1]
    return defaultdict(int, **{k[0]:v for k, v in vocab.items()})

def information(vocab):
    """
    evaluate the shannon information contained in the
    distribution of appearance rate of a given word in
    a given bin of the text.
    """
    info = defaultdict(int)
    for k, v in vocab.items():
        p = np.array(list(v.values()))
        p = p/np.sum(p)
        i = - np.sum(p * np.log(p))
        info[k] = i
    return info

def apply_to_reviews(method):
    """
    this method is a decorator to be used on class methods
    for the reviews object. This takes the input method,
    and applies it to each row on the self.reviews dataframe
    and returns a list.
    
    These methods should be of the form:
    
    def method(x, **kwargs):
        return ...
    """
    def new_method(self, x, **kwargs):
        return list(self.reviews.apply(partial(method, **kwargs), axis=1))
    return new_method

def percentile(df, field, by=None, set=True):
    """
    this is a function that evaluates a the percentile for a given
    *field* for an input pd.DataFrame *df*, and performs this operation
    *by* another input field. *set* indicates whether or not to set the percentile
    as a column in df. This can be boolean, or a string. If this is True, it
    will default to calling the new field (field + 'Percentile'), but if it is
    a string, it will call the new field (set).
    """
    if by is not None:
        df0 = df.copy()
        temp_field = 'PercentileTemp'
        df0[temp_field] = [np.nan] * len(df0)
        if by in df0.index.names:
            df0.reset_index(inplace=True)
        for p in df0[by].unique():
            inds = np.argsort(df0[df0[by] == p][field])
            df0.loc[df0[by]==p, temp_field] = (inds/np.max(inds))
        pcentile = list(df0[temp_field])
    else:
        inds = np.argsort(df[field])
        pcentile = inds/np.max(inds)
    if set is None:
        set = field + 'Percentile'
    if not(isinstance(set, (bool, str))):
        raise IOError('set input must be boolean or a string')
    if isinstance(set, bool):
        if not(set):
            return pcentile
        else:
            set = field + 'Percentile'
    df[set] = pcentile
    return


class Reviews(object):
    """
    this reviews object is used to hold and manipulate
    messages in a reviews that has been imported from
    google hangouts. This includes methods for text analysis
    in order to determine modes of reviews, and to determine
    messages that convey more information than others
    """
    required_columns = ['Time', 'Message']
    optional_columns = ['WordCount', 'LinkCount', 'DayOfWeek', 'Tokenized', 'TokenizedNSW', 'MWTokenized', 'MWTokenCount']
    optional_columns2= ['Entropy']
    day_of_week      = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    standard_contractions = './data/contractions.txt'
    add_to_stopwords = ['also', 'computer', 'laptop', 'got', 'get', 'use', 'used', 'using',
                        'one', 'two', '34', 'would']
    
    def __init__(self, reviews, vocab=None,
                 verbose=True, 
                 contractions=None, n_gram_n_max=2,
                 phrasing=True, frequency_minimum=75,
                 con=None, stopwrds=None):
        """
        inputs:
        *reviews* (pd.DataFrame) a dataframe with an integer index and columns:
            (columns in parentheses are optional - if they are not input, then they will be
             evaluated using class methods)
            Time         time of message
            Message      the text included in a given message
            (WordCount)  number of words in a message (should I change this to number of tokens?)
            (LinkCount)  number of links in the message (this includes urls, stickers, photographs, ...)
            (DayOfWeek)  day of the week as a string
            (Tokenized)  the message with text that has been cleaned and tokenized according to a given method.
            (MWTokenized)  the message with text that has been tokenized with an n-gram tokenizer with the stop words removed
        *verbose* (Boolean) whether or not to output text explaining the operations beinf performed
        *contractions* (pd.DataFrame) a dictionary of substitutions to make in the simplified text 
            (mostly contractions, common misspellings, canadian to american english, common emphasis removed ...)
        options:
        *n_gram_n_max* (int) maximum length for n_grams to search for
        *phrasing* (Boolean) whether or not to ignore phrases that include the beginning or ending o phrases
        frequency_minimim (int) minimum frequency required to include an n_gram in a tokenizer.
        *con* (sql connection) a connection to an sql database for saving the results.
        """
        modified = False

        if verbose: print('checking input format ...')
        # throw an error if the input is not a pd.DataFrame
        if not isinstance(reviews, pd.DataFrame):
            raise IOError('reviews input must be a pandas dataframe')
        
        # determine if any columns are missing, if so, throw an error
        missing_columns = [req_col for req_col in self.required_columns
                           if req_col not in reviews.columns]
        if len(missing_columns) > 0:
            raise IOError('the input dataframe is missing the following required columns: ' 
                          + ', '.join(missing_columns))
        
        # extract the columns that we would like to save:
        excess_columns = [c for c in reviews.columns if
                          c not in (self.required_columns + 
                                    self.optional_columns +
                                    self.optional_columns2)]
        
        # if len(excess_columns) > 0:
        #     print('removing unneeded columns')
        #     for ex_col in excess_columns:
        #         if verbose: print('removing column: ' + ex_col)
        #         reviews.drop(ex_col, axis=1, inplace=True)
                
        self.reviews = reviews
        self.verbose      = verbose
        self.options      = dict(n_max=n_gram_n_max, phrasing=phrasing, 
                                 threshold=frequency_minimum, verbose=verbose)
        self.con = con
        if stopwrds is None:
            self.stopwords = stopwords.words('english') + self.add_to_stopwords
        else:
            self.stopwords = stopwrds
        
        # import standard contractions if not input directly
        if contractions is None:
            if verbose: print('importing standard contractions dictionary')
            contractions = pd.read_csv(self.standard_contractions)
            contractions = {k.lower(): v.lower() for k, v in zip(contractions.Contraction,
                                                                 contractions.Expanded)}
        self.contractions = contractions
        
        # create any of the missing optional fields:
        for op_col in self.optional_columns:
            if op_col not in self.reviews.columns:
                modified=True
                self.generate_field(op_col)
                self.save_temp()
        
        # evaluate vocabulary if not already evaluated
        if vocab is None:
            if verbose: print('creating vocabulary list')
            modified=True
            vocab = self.obtain_mw_vocab()
        self.vocab = vocab
        
        # evaluate vocabulary appearance entropy if not already present
        if 'Entropy' not in self.vocab.columns:
            modified=True
            self.obtain_vocab_entropy()
        
        if 'Entropy' not in self.reviews.columns:
            modified=True
            self.generate_field('Entropy')

        if 'AvgRating' not in self.vocab.columns:
            if verbose: print('evaluating average rating for each word')
            modified=True
            self.vocab['AvgRating'] = self.obtain_average_rating()
                
        if verbose: print('reviews construction completed.')
        if self.con is not None and modified:
            self.to_sql(self.con)
        else:
            if verbose: print('please save the object with the *to_sql* method')
            
    def __repr__(self):
        return self.reviews.__repr__()

    def reset(self):
        self.reviews.drop(['Tokenized','TokenizedNSW',
                           'MWTokenized','MWTokenCount',
                           'Entropy'], axis=1, inplace=True)
        if self.con is not None:
            self.to_sql(self.con)
            self.con.execute('drop table "Vocab";')
        return
    
    def _repr_html_(self):
        return self.reviews._repr_html_()

    def save_temp(self):
        self.reviews.to_pickle('reviews_temp.pkl')
        return
                
    def generate_field(self, column):
        """
        this function generates missing optional columns
        """
        if self.verbose: print('generating column ' + column)
        if   column == 'DayOfWeek':
            v = self.obtain_day_of_week()
        elif column == 'WordCount':
            v = self.obtain_word_count()
        elif column == 'LinkCount':
            v = self.obtain_link_count()
        elif column == 'Tokenized':
            v = self.obtain_tokenized_message()
        elif column == 'TokenizedNSW':
            v = self.obtain_tokenized_message_wo_stop_words()
        elif column == 'MWTokenized':
            try:
                self.mwe_tokenizer
            except:
                # creating a multiword tokenizer.
                self.mwe_tokenizer = self.create_multiword_tokenizer(**self.options)
            v = self.obtain_mwtokenized_message()
        elif column == 'MWTokenCount':
            v = self.obtain_mwtoken_count()
        elif column == 'Entropy':
            v = self.obtain_entropy()
        else:
            raise IOError('column ' + column + ' is note recognized.')
        self.reviews[column] = v
        return
        
    def obtain_day_of_week(self):
        return list(self.reviews.apply(lambda x : 
                                       self.day_of_week[x.Time.dayofweek],
                                       axis=1))
        
    def obtain_word_count(self):
        return list(self.reviews.apply(lambda x :
                                       len(x.Message.split()),
                                       axis=1))
        
    def obtain_link_count(self):
        return list(self.reviews.apply(lambda x :
                                       len(re.findall(
                                       'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                                       x.Message)), axis=1))
        
    def obtain_tokenized_message(self, phrasing=True):
        return list(self.reviews.apply(lambda x :
                    ' '.join(self.tokenize(x.Message, phrasing=phrasing)),
                    axis=1))

    def obtain_tokenized_message_wo_stop_words(self):
        return list(self.reviews.apply(lambda x :
                    ' '.join([w for w in x.Tokenized.split()
                              if w not in self.stopwords]),
                    axis=1))
    
    def obtain_mwtokenized_message(self, phrasing=True):
        return list(self.reviews.apply(lambda x :
                    ' '.join(self.mwe_tokenizer.tokenize(
                    x.TokenizedNSW.split())),
                    axis=1))
    
    def obtain_mwtoken_count(self):
        return list(self.reviews.apply(lambda x :
                                       len(x.MWTokenized.split()),
                                       axis=1))
    
    def obtain_entropy(self):
        df = self.vocab.set_index(['Word'])
        return list(self.reviews.apply(lambda x :
                                       np.sum([
                        df.DiffEntropy[word]
                        for word in x.MWTokenized.split()
                        if word != '.'
                    ]), axis=1))
    
    @staticmethod
    def read_sql(conn, table='Messages', verbose=True, **kwargs):
        """
        create object from an sqlite database
        """
        if verbose: print('importing reviews from sql database')
        reviews = pd.read_sql('SELECT * FROM "Messages"', conn, parse_dates=['Time'])
        if 'index' in reviews.columns:
            reviews.set_index('index', inplace=True)
        try:
            vocab = pd.read_sql('SELECT * FROM "Vocab"', conn)
            if 'index' in vocab.columns:
                vocab.set_index('index', inplace=True)
        except:
            vocab = None
        return Reviews(reviews, vocab=vocab, verbose=verbose, con=conn, **kwargs)
    
    def to_sql(self, conn):
        if self.verbose: print('saving reviews to sql database')
        self.reviews.to_sql('Messages', conn, if_exists='replace')
        self.vocab.to_sql('Vocab', conn, if_exists='replace')
        return
    
    def textify(self, df, joiner=' . ', by='Message'):
        """
        given input pd.DataFrame df, convert the
        messages into a single stream of text
        where adjecent messages are joined by the
        *joiner*
        """
        return joiner.join(list(df[by]))
    
    def decontract(self, text):
        """
        given an input string, "text", replace contractions
        in the keys of dictionary self.contractions. this assumes
        that text protects urls with brackets, <http://...>, and
        is lowercase.
        """
        # for each contract, perform a replacement operation
        
        for k, v in self.contractions.items():
            text = re.sub(r'([\W]' + k + '[\W])', ' ' + v + ' ', text)
            # text = re.sub(r'((?!<[\S]*)[\W]' + k + '[\W](?![\S]*>))', ' ' + v + ' ', text)
            # text = re.sub(r'((?!<[\S]*)' + k + '(?![\S]*>))', v, text)
        return text
    
    def tokenize(self, text_input, phrasing=True):
        """
        takes an input *text_input* string, and parses it into a list of words.
        operations performed:
        -making all letters lowercase
        -isolating links
        -removing emphasis where possible
        -replacing contractions with uncontracted versions
        -removing punctuation.

        if phrasing=True, we leave periods in the tokenized output
        to denote where phrases end.
        """
        text = ' ' + text_input + ' '
        text = text.lower()
        # remove links and replace them with <link>
        # text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 
        #        lambda x: ' <' + x.group(0) + '> ', 
        #        text)
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',' ',text)
        # de-emphasize words by removing cases where letters appear more than twice in a row
        text = re.sub(r'(?!<[\S]*)(.)\1+(?![\S]*>)', r'\1\1', text)
        # remove punctuation if possible (but not in links)
        text = re.sub(r'((?!<[\S]*)[\"\)\(\[\]\;\~\_\:\+\,\|\\\*\#\&\<\>\/\-](?![\S]*>))', r' ', text)
        #protect numerical decimal points:
        text = re.sub(r'(?<=\d)\.(?=\d)',',',text)
        # deal with the funny quotes:
        text = re.sub(r'&quot', ' inch',text)
        # replace end punctuation with period for parsing
        text = re.sub(r'((?!<[\S]*)([\.\?\!]\s?)+(?![\S]*>))+', r' . ', text)
        # treat individual emojis like words rather than characters:
        # text = re.sub("(["
        #              u"\U0001F600-\U0001F64F"  # emoticons
        #              u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        #              u"\U0001F680-\U0001F6FF"  # transport & map symbols
        #              u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        #                        "])",r' \1 ',text)
        # replace contractions (but not in links)
        text = self.decontract(text)
        # remove punctuation if possible (but not in links)
        text = re.sub(r'((?!<[\S]*)[\'\=](?![\S]*>))', r' ', text)
        
        #phrases = [re.sub('( and | or | because | but )+', r'. ', phrase)
        #           for phrase in phrases]
        if not(phrasing):
            text = re.sub(r'((?!<[\S]*)[\.]+(?![\S]*>))+', r' ', text)
        return text.split()

    def mw_tokenize(self, text):
        text = self.tokenize(text, phrasing=True)
        text = [word for word in text
                if word not in self.stopwords]
        text = self.mwe_tokenizer.tokenize(text)
        return text
    
    def create_multiword_tokenizer(self, joiner=' . ', n_max=5, phrasing=True,
                                   threshold=50, use_threshold=True, **kwargs):
        """
        create a multiword tokenizer in the reviews
        options:
        joiner (string) string to use to connect messages within the reviews.
        n_max (int) max number of words to include in the n_grams
        phrasing (Boolean) whether or not to terminate allowed n_grams with
            the end of messages or with punctuation.
        threshold (int) minimum frequency to include an n_gram in the multiword_tokenizer
        use_threshold (Boolean) whether or not to use a minimum frequency threshold.
        """
        mwe_tokenizers = {}
        if self.verbose: print('creating multiword tokenizer')
        if 'MWTokenized' not in self.reviews.columns:
            if 'Tokenized' not in self.reviews.columns:
                self.reviews['Tokenized'] = self.generate_field('Tokenized')
            if 'TokenizedNSW' not in self.reviews.columns:
                self.reviews['TokenizedNSW'] = self.generate_field('TokenizedNSW')
            phrased_tokens = ' . '.join(self.reviews.TokenizedNSW).split()
            n_gram_vocab   = combined_n_grams(phrased_tokens, n_max=n_max,
                                              threshold=threshold, 
                                              use_threshold=use_threshold)
            # create the multi-word tokenizer:
            mwe_tokenizer = MWETokenizer(n_gram_vocab.keys())
        else:
            # convert the multiword-tokenized messages into a text
            text  = self.textify(self.reviews, joiner=joiner, by='MWTokenized')
            # get the vocabulary word list for this text
            vocab = get_vocab(text.split())
            words = vocab.keys()
            # since the multiword tokens are combined with the _ character
            # and since MWETokenizer wants tuples, use this pattern to create tuples
            PATTERN = re.compile(r'''((?:[^_<]|<[^>]*>)+)''')
            n_grams = [tuple(PATTERN.split(w)[1::2])
                       for w in words]

            mwe_tokenizer = MWETokenizer(n_grams)
                
        return mwe_tokenizer
    
    def obtain_mw_vocab(self):
        # obtain the tokens 
        tokens = self.textify(self.reviews, 
                              joiner=' ', by='MWTokenized').split()
        vocab = get_vocab(tokens)
        # create the vocabulary dataframe
        vocab = [(word, freq) 
                 for word, freq in vocab.items()]
        vocab = pd.DataFrame(vocab, columns=['Word', 'Frequency'])
        vocab.sort_values(by=['Frequency'], ascending=False, inplace=True)
        # evaluate the number of words that make up each n_gram:
        PATTERN = re.compile(r'''((?:[^_<]|<[^>]*>)+)''')
        length  = [len(PATTERN.split(w)[1::2]) for w in vocab.Word]
        vocab['Length'] = length
        return vocab
    
    def obtain_vocab_entropy(self):
        if self.verbose: print('evaluating entropy relative to expected poisson entropy.')

        i, n = self.measured_information(self.reviews)
        # create the entropy fields
        self.vocab['PoissonEntropy'] = self.prior_information(self.vocab, n)
        self.vocab['Entropy'] = list(self.vocab.apply(lambda x:
                                     i[x.Word], axis=1))
        # diff entropy is a metric of deviations from Poissonian entropy
        # which gives me a good metric for information carried by a word
        self.vocab['DiffEntropy'] = self.vocab['PoissonEntropy'] - self.vocab['Entropy']
        self.vocab.sort_values(by='DiffEntropy', ascending=False, inplace=True)
        # here is a metric for long characteristic phrases uttered
        self.vocab['LengthLogFrequency'] = self.vocab.Length * np.log(self.vocab.Frequency)

        percentile(self.vocab, 'DiffEntropy', set='DiffEntropyPercentile')
        # create a percentile ranking by participant for LengthLogFrequency.
        percentile(self.vocab, 'LengthLogFrequency', set='LengthLogFrequencyPercentile')
        self.vocab.reset_index(drop=True, inplace=True)
        return
    
    def measured_information(self, reviews, grouping=None):
        tokens = [t for t in (' '.join(reviews.MWTokenized)).split()
                  if t != '.']
        vocab = defaultdict(lambda : defaultdict(int))
        if grouping is None:
            grouping = int(np.sqrt(len(tokens)))
        N = len(tokens)//grouping
        for i in range(N):
            group = tokens[i*grouping : (i + 1)*grouping]
            subvocab = get_vocab(group)
            for k, v in subvocab.items():
                vocab[k][v] += 1
        for k, v in vocab.items():
            M = sum(v.values())
            v[0] = N - M
        return information(vocab), N

    def prior_information(self, vocab, N):
        mu = vocab.Frequency/np.array(N)
        unique_mu = np.unique(mu)
        prior = {m: e for m, e in
                 zip(unique_mu, poisson.entropy(unique_mu))}
        return [prior[f] for f in mu]

    def relative_information(self, grouping=None):
        post_info, N  = self.measured_information(self.reviews, grouping=grouping)
        p_vocab   = self.vocab
        post_info = [post_info[word] for word in p_vocab.Word]
        prior_info    = self.prior_information(p_vocab, N)
            
    def find_instances(self, term, field='Tokenized', pad=' '):
        return self.reviews.ix[
            [i for i in range(len(self.reviews))
            if (pad + term + pad) in self.reviews[field][i]]
        ]

    def obtain_average_rating(self):
        vocab = defaultdict(lambda : [])
        for rating, text in zip(self.reviews.overall, 
                                self.reviews.MWTokenized):
            words = text.split()
            for w in words:
                vocab[w].append(rating)
        avg_rating = {w:np.mean(v) for w, v in vocab.items()}
        avg_rating = [avg_rating[w] for w in self.vocab.Word]
        return avg_rating

    def remove_stopwords(self, field='Tokenized', stop_words=None):
        if stop_words is None:
            stop_words = stopwords.words('english')
        new_message = []
        for message in self.reviews[field]:
            new_message.append([])
            for word in message.split():
                if word not in stop_words:
                    new_message[-1].append(word)
            new_message[-1] = ' '.join(new_message[-1])
        return new_message