from review_analysis import FaultFindr
import json
import yaml
import pickle
from itertools import product
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from fuzzywuzzy import process

MODEL_DIRECTORY = './faultfindr/models/'
DATABASE_CONFIG = './faultfindr/faultfindr_database.yaml'

def load(filename):
    with open(filename) as f:
        return pickle.load(f)

class Laptops(object):
    """
    this is an object that contains a list of all of the laptops
    and enables searching for subsets

    methods:
    get_query_matches: given an input search query, provide a list of asins (amazon product ids)
                        that match the query.
    search results:    given an input search query, provide a list of dictionaries containing
                        details of the laptops match the query.
    get_details:       given an input asin (amazon product id), provide the properties of the laptop
                        with that asin as a dictionary.
    """
    
    conn_config_file = DATABASE_CONFIG
    
    def __init__(self, 
                 conn=None, 
                 min_n_reviews=1,
                 max_n_results=9):
        """
        inputs:
        conn: either a database connection, or a string to a config file with a database connection string
              (if not provided, creates the default database connection)
        min_n_reviews: (int) the minimum number of reviews that a product must have to load it into the database
        max_n_results: (int) the maximum number of results to display upon a given for a certain search query
        """
        
        self.conn    = self._get_conn(conn)
        self.min_n_reviews = min_n_reviews
        self.max_n_results = max_n_results
        
        # load a table with laptop properties into a dataframe in memory:
        self.laptops = pd.read_sql(""" SELECT asin, title, reviews, "imUrl" FROM "Laptops" 
                                       WHERE reviews > """ + str(self.min_n_reviews) + """;""", self.conn)
        self.laptops.sort_values(by='reviews', ascending=False, inplace=True)
        
    def _get_conn(self, conn):
        """
        rectify various possible types of inputs for the database connection conn

        inputs:
        conn: either a database connection, or a string to a config file with a database connection string
              (if not provided, creates the default database connection)
        """
        if conn is None or isinstance(conn, str):
            if isinstance(conn, str):
                self.conn_config_file = conn
            with open(self.conn_config_file, 'r') as f:
                config_file = yaml.load(f)
            conn = create_engine(config_file['connection_string'])
        return conn
    
    def get_query_matches(self, query):
        """
        This is a simple laptop search function. This ranks search results
        by whether or not they include all of the search terms, and by
        number of reviews for a given product.
        """
        results = np.argsort(
            self.laptops.apply(
                lambda x: -int(all(
                        [word in str(x.title).lower()
                          for word in query.lower().split()])) * x.reviews,
                        axis=1))

        if len(results) > self.max_n_results:
            results = results[:self.max_n_results]
            
        asins = [self.laptops['asin'].iloc[result] for result in results]
        return asins
    
    def search_results(self, query):
        """
        This function returns search results if there is a search
        query, but otherwise returns the products with the most reviews
        and displays at most self.max_n_results number results
        """
        if query is None:
            results = self.laptops
        else:
            asins = self.get_query_matches(query)
            results = self.laptops[
                            self.laptops.asin.isin(asins)
                      ].sort_values(by='reviews', ascending=False)
        if len(results) > self.max_n_results:
            results = results.iloc[:self.max_n_results]
        return [{k: str(results[k][i]) for k in results.columns}
                for i in results.index]
        
    def get_details(self, asin):
        """
        get the properties with an input asin (amazon product id as a string) and
        provide the results in a dictionary
        """
        if asin is None:
            return None
        laptop = self.laptops[self.laptops.asin == asin]
        return {k:laptop[k].iloc[0] for k in list(laptop.columns)}



class ProductResults(object):
    """
    This is an object used to obtain and to perform and hold results of the failure analysis model

    methods:
    get_review_number_barchart_data:    obtain json formatted data to feed into the d3 barchart
                                        showing number of reviews over time
    get_pie_chart_discussion_data:      obtain a dictionary of json formatted data to feed into
                                        the d3 pie charts showing the distribution of discussion topics
                                        within each topic.
    get_related:                        obtain a list of dictionaries of laptop properties for a series
                                        of "related products" as defined by Amazon.
    """
    
    # There are a set of pre-defined models that I trained for identifying reviews with discuss
    # certain topics. However, for the website, I do not want to show as fine grained information
    # as I have, so I am grouping together the results of sets of models under new names with
    # the following mapping:
    field_groups ={'failure'            : ['failure'],
                   'internal-hardware'  : ['motherboard_gpu_memory_processor', 'hard_drive'],
                   'battery-charger'    : ['battery', 'charger'],
                   'keyboard-trackpad'  : ['keyboard', 'trackpad', 'ports', 'disk_drive'],
                   'audio-video'        : ['screen', 'camera', 'audio'],
                   'software'           : ['software', 'operating_system_bios'],
                   'build'              : ['build_design_quality', 'cooling_system_fan_noise'],
                   'performance'        : ['speed_power_responsive', 'freeze_crash_boot_issue'],
                   'customer-service'   : ['customer_service_returns'],
                   'internet'           : ['wifi_bluetooth_internet'],
    }
    
    # this loads the default classification model from a pickle object:
    model_filename = MODEL_DIRECTORY + 'simple_naive_bayes_model_275_features.pkl'
    model = load(model_filename)

    # this is a path to the config file containing an address string
    # for the default database connection
    conn_config_file = DATABASE_CONFIG

    # this is a path to a pickled trained sentence tokenizer.
    sentence_tokenizer_filename = MODEL_DIRECTORY + 'sentence_tokenizer.pkl'

    
    def __init__(self, asin,
                 conn=None,
                 model=None,
                 n_max_examples=6,
                 time_format='%b %y',
                 n_max_results=9,
                 group_examples_by_author=True,
                 sentence_tokenizer=None,
                 sentence_mapping=False):
        """
        inputs:
        asin: (string) an amazon product ID string 
        conn: a database connection or a yaml config file with connection string
        model: (string) a path to a .pkl file containing a model to use to evaluate the reviews
        n_max_examples: (int) the maximum number of examples to provide for each topic model
        time_format: (string) a time formatting string for the x axis of the bar chart showing
                    number of reviews over time.
        n_max_results: (int) the maximum number of "similar products" to display for a give product.
        group_examples_by_author: (boolean) whether or not to group sentence examples from the same
                                    author into the same line.
        sentence_tokenizer: a sentence tokenizer object, or a filename for a sentence tokenizer
        sentence_mapping: (boolean) whether or not to try to make the tokenized text to the untokenized text.
        """

        self.asin           = str(asin)
        self.n_max_examples = n_max_examples
        self.time_format    = time_format
        self.n_max_results  = n_max_results
        self.group_examples_by_author = group_examples_by_author
        self.sentence_mapping         = sentence_mapping
        
        # we have a default config file that we can pass in to open a database connection
        # this function either opens that connection, opens a connection for an input yaml
        # file, or grabs an input database connection.
        self.conn = self._get_conn(conn)
        
        # we have a preloaded model, but this opens a new model if a .pkl filename is passed:
        self._get_model(model)

        # this is loading of a pretrained sentence tokenizer model
        self._get_sentence_tokenizer(sentence_tokenizer)
        
        # grab the reviews data and break it up into sentences:
        self.reviews   = self._get_reviews()
        self.sentences = self._get_sentences()
        
        # perform the analysis:
        self.ranked_examples         = self._get_ranked_examples()
        self.formatted_examples      = self._get_top_formatted_examples()
        self.review_rates            = self._get_review_rates()
        self.discussion_distributions= self._get_discussion_distributions()

        if self.sentence_mapping:
            self.formatted_examples = self._fuzzy_match_examples()
        
        if self.group_examples_by_author:
            self.formatted_examples = self._aggregate_examples_by_same_reviewer()
        
    def __len__(self):
        return len(self.reviews)
    
    def _get_conn(self, conn):
        # conn: either a database connection, or a string to a config file with a database connection string
        #       (if not provided, creates the default database connection)
        if conn is None or isinstance(conn, str):
            if isinstance(conn, str):
                self.conn_config_file = conn
            with open(self.conn_config_file, 'r') as f:
                config_file = yaml.load(f)
            conn = create_engine(config_file['connection_string'])
        return conn
    
    def _get_model(self, model):
        if (model is not None and
            model != self.model_filename):
            self.model_filename = model
            self.model      = load(self.model_filename)

    def _get_sentence_tokenizer(self, sentence_tokenizer):
        if (sentence_tokenizer is not None and
            sentence_tokenizer != self.sentence_tokenizer_filename):
            self.sentence_tokenizer_filename = sentence_tokenizer
        self.sentence_tokenizer = load(self.sentence_tokenizer_filename)
        
    def _get_reviews(self):
        """
        obtain all of the reviews that correspond to product ID self.asin
        from the table "Messages" using the database connection self.conn
        """
        reviews = pd.read_sql("""SELECT * FROM "Messages"
                                 WHERE asin = '""" + self.asin + "';", self.conn)
        return reviews
    
    def _get_sentences(self, method="manual"):
        """
        obtain the mapping between tokenized and untokenized sentences this can be done
        with the following methods:
        method='cached' - grab the mapping from a database
        method='manual' - evaluate the mapping on the fly
        """
        if method == "cached":
            sentences = pd.read_sql("""SELECT messages, tokenized FROM "SentenceMapping"
                                       WHERE asin = '""" + str(asin) + """';""", self.conn)
        elif method == "manual":
            
            sentify = lambda review : [sentence for sentence in review.split('.')
                                       if sentence not in ['', ' ', '.']]
            
            sentences = [(index, tokenized, messages) 
                         for index, (tokenized_review, messages_review)
                         in enumerate(zip(self.reviews.MWTokenized,
                                          self.reviews.Tokenized))
                         for tokenized, messages in zip(
                             sentify(tokenized_review), sentify(messages_review)
                         )]

            sentences = pd.DataFrame(sentences, columns=['index', 'tokenized', 'messages'])
        return sentences
    
    def _get_ranked_examples_for_field_set(self, fields):
        """
        feed the extracted sentences into each predictive model
        and obtain the indices that correspond to positive results
        ranked by probability a sentence having been correctly identified
        for one of the groups of model fields included in self.field_groups
        """
        
        if not(isinstance(fields, (list, tuple))):
            fields = [fields]
            
        # initialize probability array
        p = np.zeros(len(self.sentences))
        for field in fields:
            p_i = self.model[field].predict_proba(
                                self.sentences['tokenized']
            )[:,1]
            
            # set those probabilities to zero for those which are not
            # above threshold
            p_i = np.array([
                    p_val if (p_val > self.model[field].threshold)
                    else 0.0
                    for p_val in p_i
                ])
            
            # rank probabilities by the max probability for a sentence over all fields
            p = np.max([p, p_i], axis=0)
        
        inds = list(reversed(list(np.argsort(p))))
        return [i for i in inds if p[i] > 0]
    
    def _get_ranked_examples(self):
        """
        obtain the indices of the sentences that return positive
        results in the predictive models for all models
        """
        return [
            [name, self._get_ranked_examples_for_field_set(fields)]
            for name, fields in self.field_groups.items()
        ]
    
    def _get_top_formatted_examples(self):
        """
        given the ranked output indices of sentences that
        produced positive model results, provide the corresponding
        sentences and truncate that list of sentences if that list
        exceeds a length of self.n_max_examples.
        """
        formatted_examples = []
        for name, inds in self.ranked_examples:
            item = [name, [{'message': self.sentences['messages'][i],
                            'author' : self.reviews.iloc[
                                       self.sentences['index'][i]
                            ].reviewerName,
                            'index'  : self.sentences['index'][i]}
                                                            for i in inds]]
            if self.n_max_examples is not None:
                if len(item[1]) > self.n_max_examples:
                    item[1] = item[1][:self.n_max_examples]
            formatted_examples.append(item)
        return formatted_examples

    def _fuzzy_match_examples(self):

        formatted_examples = []
        for name, examples in self.formatted_examples:
            new_examples = []
            for example in examples:
                print(example['index'])
                print(self.reviews.Message.iloc[example['index']])
                messages = self.sentence_tokenizer.tokenize(
                                        self.reviews.Message.iloc[example['index']]
                                        )
                new_examples.append(
                    {'message' : process.extract(example['message'], messages, limit=1)[0][0],
                     'author'  : example['author']}
                    )
            formatted_examples.append([name, new_examples])
        return formatted_examples

    def _aggregate_examples_by_same_reviewer(self):
        """
        given the self.formatted_examples, group together examples
        which have the same reviewer for a clearer display of the information

        this also removes redundant comments
        """
        formatted_examples = []
        for name, examples in self.formatted_examples:
            new_examples = []
            authors      = []
            for example in examples:
                author = example['author']
                message = example['message']
                if author in authors:
                    if message not in new_examples[authors.index(author)]['message']:
                        new_examples[authors.index(author)]['message'].append(message)
                else:
                    authors.append(author)
                    new_examples.append({'message': [message],
                                         'author' : author})
            new_examples = [{'message': ' ... '.join(item['message']),
                             'author' : item['author']} for item in new_examples]
            formatted_examples.append([name, new_examples])
        return formatted_examples


    
    def _get_review_rates(self):
        """
        outputs a tuple of the form (number of reviews with positive results,
                                     total number of reviews,
                                     percentage (rounded to nearest integer) of positive results)
        for each of the models.
        """
        review_rates = {}
        total = float(len(self))
        for field, inds in self.ranked_examples:
            number = len(set(inds))
            review_rates[field] = (number, int(total), int(100 * number / total))
        return review_rates
    
    def _get_discussion_distributions(self):
        """
        outputs the number of sentences that produced a positive result for a model
        for the set of reviews that produced positive results for another model.
        """
        discussion_distributions = {field1:{field2:0 for field2 
                                            in self.field_groups.keys()
                                            if field2 != field1}
                                    for field1
                                            in self.field_groups.keys()}
        
        for example1, example2 in product(self.ranked_examples,
                                          self.ranked_examples):
            field1, inds1 = example1
            field2, inds2 = example2
            
            if field1 != field2:
                discussion_distributions[field1][field2] = (
                    len([i for i in inds2
                         if i in inds1])
                )
        return discussion_distributions
    
    def get_review_number_barchart_data(self):
        """
        get json data for a d3 bar chart showing the number
        of reviews over time
        """
        
        n  = self.reviews.set_index('Time').groupby(pd.TimeGrouper('Q')).aggregate(len)
        t = [time.strftime((self.time_format)) for time in n.index]
        y = n.overall

        return json.dumps([{"x": time,
                            "y": number} for time, number
                            in zip(t, y)])
    
    def get_pie_chart_discussion_data(self):
        """
        get json data for a series of d3 pie charts showing
        the distribution of discussion topics within
        each topic.
        """
        return {
            field: json.dumps([{"value":y,
                                "label":x} for x, y in distribution.items()])
            for field, distribution in self.discussion_distributions.items()
        }

    def get_related(self):
        """
        get a list of dictionaries of properties of laptops that are judged by amazon to be
        "similar products"
        """
        related = pd.read_sql("""
                        SELECT a.asin, a.title, "imUrl", a.reviews FROM "Laptops" a JOIN "Related" b
                        ON a.asin = b.related_asin
                        WHERE b.asin = '""" + str(self.asin) + """'
                        ORDER BY a.reviews DESC
                        LIMIT """ + str(self.n_max_results) + ";", self.conn)
        related = [{k: related[k].iloc[i] for k in related.columns}
                                      for i in range(len(related))]
        return related