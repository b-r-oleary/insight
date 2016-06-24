# FaultFindr

FaultFindr is a web app located at [http://faultfindr.xyz]() that uses a predictive model to identify laptop failures in Amazon product reviews in order to create a summarization of failure modes for various laptops and to estimate failure rates.  This web app is built using Flask, Bootstrap, and D3 that is hosted on AWS. On the backend, text processing is done using regular expressions, nltk, and textblob, and the predictive models are based on sklearn. The web application relies on a PostgreSQL database that is not supplied here. The application can be run with the `faultfindr.py` script.

### Structure

- `faultfindr.py`: main application
- `faultfindr`: application module
    - `__init__.py`
    - `views.py`: primary flask function for generating web. pages
    - `review_processing.py`: script that contains some objects and functions used for text cleaning and processing.
        - `Reviews`: Object that imports reviews, cleans the text, removes stopwords, identifies n-grams above a certain frequency threshold, and tokenizes the text with respect to these specified n-grams.
    - `review_analysis.py`: a script containing some objects that house the predictive models.
        - `ClassificationModel`: an object that wraps a binary sklearn classification model, and contains a bag of words vectorizer, and includes some methods for evaluating the performance of the model and plotting.
        - `FaultFindr`: an object that trains and houses an array of `ClassificationModel` objects.
    - `product_results.py`: a script containing some objects that are used to obtain information about Laptops, and to obtain results of the failure mode analysis for each set of reviews corresponding to a given laptop for the web application.
        - `Laptops`: houses a list of all of the laptops, and includes a search function for obtaining a list of laptops that match a given search query.
        - `ProductResults`: houses the set of reviews for a given laptop, applies a predictive model for classifying each sentence within each review, and has methods to generating json data for feeding into D3 plots.
    - `models`: a directory with pickled pretrained `FaultFindr` models
    - `static`: a directory that contains mostly standard bootstrap files and the agency bootstrap template for creating the webpage
    - `templates`: a directory containing `index.html` which is a flask/jinja html-template for generating the webpage with flask.
    
### Database

Here is a rough overview of the database structure for reference, since access to the database is not provided.

#### Tables:

- `Laptops`: a list of laptops with properties
    - `asin` (str) amazon product review ID
    - `title` (str) name of the laptop
    - `description` (str) description of the laptop
    - `price` (float) price in USD
    - `brand` (str) brand of the laptop
    - `related` (int) number of related items
    - `imUrl` (str) url to the image of the object
    - `refurbished` (boolean) whether or not it is refurbished
    - `reviews` (int) number of reviews
    - `screen_size` (double) screen size in inches
- `Messages`: a table of all of the reviews
    - `asin`: (str) amazon product review corresponding to the laptop
    - `overall`: (int) amazon star rating
    - `Time`: (timestamp) time of review
    - `summary`: (str) title of the review
    - `Message`: (str) text of the review combined with the title
    - `reviewerName`: (str) name of the reviewer
    - `Tokenized`: (str) cleaned text tokenized by unigrams
    - `TokenizedNSW`: (str) cleaned text tokenized by unigrams with stopwords removed
    - `MWTokenized`: (str) cleaned text tokenized by n-grams with stopwords removed.
    
- `Vocab`: a table of the n-grams in the n-gram vocabulary for tokenization

    - `Word`: (str) text of the word
    - `Frequency`: (str) number of occurences in the whole corpus
    - `Length`: (str) number of unigrams in the n-gram
- `Related`: Table of mappings between laptops and related laptops (as specified by Amazon)
    - `asin`: (str) amazon product id
    - `related_asin`: (str) amazon product id for a related item
- `TrainSentences`: a table with the training data.
    - `sentence`: (str) n-gram tokenized sentence with stop words removed
    - `readable`: (str) untokenized text.
    - Columns with Outcomes (all boolean): `audio`, `battery`, `build_design_quality`, `camera`, `charger`, `cooling_system_fan_noise`, `cost`, `customer_service_returns`, `disk_drive`, `failure`, `freeze_crash_boot_issue`, `hard_drive`, `keyboard`, `motherboard_gpu_memory_processor`, `operating_system_bios`, `ports`, `screen`.

