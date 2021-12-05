import csv, re
import math
import json
import string
import argparse
import codecs
import spacy
from tqdm import tqdm
from statistics import mode
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics
import llr as llr
from collections import Counter
from sklearn.model_selection import train_test_split
from spacy.lang.en import English
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from project_fns import *

# Convenient for debugging 
from traceback_with_variables import activate_by_import

# Hard-wired variables
stopwords_file     = "data/mallet_en_stoplist.txt"
input_csv_reviews   = "data/Kindle_Store_5.csv"
input_json_reviews = "data/Kindle_Store_5.jsonl"
negative_reviews          = "data/negative_reviews.txt"
positive_reviews          = "data/positive_reviews.txt"
moderate_reviews          = "data/moderate_reviews.txt"
topN_to_show       = 100
cutoff = 2500 # number of reviews to convert to jsonl from csv. Total reviews: ~2.2 Million

# Read a set of stoplist words from mallet_en_stoplist.txt, assuming it contains one word per line
def load_stopwords(filename):
    stopwords = []
    with open(filename, "r") as f:
        for line in tqdm(f):
            line = re.sub(r"[\n\t]*", "",line)
            stopwords.append(line)
    #print("---STOPWORDS----\n",stopwords,"\n----------")
    return set(stopwords)

def csv_to_jsonl(infile, cutoff):
    csvfile = open(infile, 'r')
    jsonfile = open('data/Kindle_Store_5.jsonl', 'w')
    fieldnames = ("ReviewerID","ProductID","Review","Feedback")
    reader = csv.DictReader(csvfile, fieldnames)
    for i, row in tqdm(enumerate(reader)):
        json.dump(row, jsonfile)
        jsonfile.write('\n')
        if i == cutoff:
            break

# Creating a list of lines and list of their corresponding labels
def read_and_clean_lines(infile):
    print("\nReading and cleaning text from {}".format(infile))
    lines = []
    labels = []
    with open(infile,'rt') as f:
        f = f.readlines()[1:]
        for line in tqdm(f):
            pass
            line_dict = json.loads(line)             
            regex = re.compile('[ \t\n\r\f\v]')
            line_dict['Review'] = regex.sub(' ',line_dict['Review'])
            line_dict['Review'] = re.sub(' +',' ',line_dict['Review'])
            #line = line_dict['Feedback']+"\t"+line_dict['Review']
            line = line_dict['Review']
            lines.append(line)
            label = line_dict['Feedback']
            labels.append(label)
    print("Read {} documents".format(len(lines)))
    print("Read {} labels".format(len(labels)))
    #print(lines[0],"\n------------------------\n")
    #print(labels[0],"\n------------------------\n")
    return lines, labels

# splitting lines and labels into X_train, X_test, y_train, and y_test
def split_training_set(lines, labels, test_size=0.3, random_seed=42):
    X_train, X_test, y_train, y_test = train_test_split(lines, labels, test_size=test_size, random_state=random_seed)
    print("Training set label counts: {}".format(Counter(y_train)))
    print("Test set     label counts: {}".format(Counter(y_test)))
    return X_train, X_test, y_train, y_test

# creating positive_reviews.txt, moderate_reviews.txt, and negative_reviews.txt containing lines from the training set
def write_user_reviews(lines, labels, outfile, feedback):
    train_lines = []
    print("status being written to {}".format(outfile))
    with open(outfile, "w") as f:
        i = 0
        for line in tqdm(lines):
            if feedback == labels[i]:
                f.write(line + '\n')
                train_lines.append(line)
            i+=1
    #print("-------------------length ----- ", len(train_lines))

def collect_bigram_counts(lines, stopwords, remove_stopword_bigrams = True, remove_punc=True):
    if (remove_stopword_bigrams):
        print("Collecting stopword-filtered bigram counts")
    else:
        print("Collecting bigram counts w/ no stopword-filtering")

    print("Initializing spacy")     # Initialize spacy and an empty counter
    nlp       = English(parser=False) # faster init with parse=False, if only using for tokenization
    nlp.tokenizer.rules = {key: value for key, value in nlp.tokenizer.rules.items() if "'" not in key and "’" not in key and "‘" not in key}
    counter   = Counter()
    
    for line in tqdm(lines):   # Iterate through raw text lines
        spacy_analysis  = nlp(line)     # Call spacy and get tokens
        spacy_tokens    = [token.orth_ for token in spacy_analysis]
        normalized_toks = normalize_tokens(spacy_tokens)    # Normalize 
        bigrams = ngrams(normalized_toks, 2) # Get bigrams
        if remove_punc:     # Filter out bigrams where either token is punctuation
            bigrams = filter_punctuation_bigrams(bigrams)
        if remove_stopword_bigrams: # Optionally filter bigrams that are both stopwords
            bigrams = filter_stopword_bigrams(bigrams,stopwords) 
        # Increment bigram counts
        for bigram in bigrams:
            if bigram[0]+"_"+bigram[1] in counter:
                counter[bigram[0]+"_"+bigram[1]] = counter[bigram[0]+"_"+bigram[1]] + 1
            else:
                counter[bigram[0]+"_"+bigram[1]] = 1
    return counter

# counting unigrams
def get_unigram_counts(bigram_counter, position):
    print("Collecting unigram counts")
    unigram_counter = Counter()
    for bigramstring in bigram_counter:
        bigramstring_position = bigramstring.split('_')[position]
        if bigramstring_position in unigram_counter:
            unigram_counter[bigramstring_position] = unigram_counter[bigramstring_position] + 1
        else:
            unigram_counter[bigramstring_position] = 1
    return unigram_counter

def compute_pmi(review_bigram_counts, review_unigram_w1_counts, review_unigram_w2_counts):
    print("Computing PMI")
    pmi_values = Counter()
    for bigramstring in tqdm(review_bigram_counts):
        bigramstring_w1 = bigramstring.split('_')[0]
        bigramstring_w2 = bigramstring.split('_')[1]
        pmi_values[bigramstring] = pmi(bigramstring_w1, bigramstring_w2, review_unigram_w1_counts, review_unigram_w2_counts, review_bigram_counts)
    return pmi_values

def pmi(word1, word2, review_unigram_w1_counts, review_unigram_w2_counts, bigram_freq):
    prob_word1 = review_unigram_w1_counts[word1] / float(sum(review_unigram_w1_counts.values()))
    prob_word2 = review_unigram_w2_counts[word2] / float(sum(review_unigram_w2_counts.values()))
    prob_word1_word2 = bigram_freq["_".join([word1, word2])] / float(sum(bigram_freq.values()))
    return math.log(prob_word1_word2/float(prob_word1*prob_word2),2) 

def compute_llr(positive_counts, negative_counts):
    llr_values = Counter()
    llr_values = llr.llr_compare(positive_counts, negative_counts)
    return llr_values

def print_sorted_items(dict, n=10, order='ascending'):
    if order == 'descending':
        multiplier = -1
    else:
        multiplier = 1
    ranked = sorted(dict.items(), key=lambda x: x[1] * multiplier)
    for key, value in ranked[:n] :
        print(key, value)

def get_users(infile):
    authors = []
    with open(infile,'rt') as f:
        f = f.readlines()[1:]
        for line in tqdm(f):
            pass
            line_dict = json.loads(line)
            author = line_dict['ReviewerID']
            authors.append(author)
    print("Read {} users".format(len(np.unique(authors))))
    #print(authors[0],"\n------------------------\n")
    return np.unique(authors)

def write_user_reviews_clf(lines, labels, outfile, feedback):
    train_lines = []
    print("status being written to {}".format(outfile))
    with open(outfile, "w") as f:
        for line, label in tqdm(zip(lines, labels), total = len(labels)):
            if label == feedback:
                f.write(line + '\n')
                train_lines.append(line)
    return(train_lines)

def read_and_clean_lines_classifier(infile):
    print("\nReading and cleaning text from {}".format(infile))
    d = dict()
    labels = []
    with open(infile,'rt') as f:
        f = f.readlines()[1:] # skip the header line
        for line in tqdm(f):
            #user = jsonString['ReviewerID']
            jsonString = json.loads(line)
            jsonString['Review'] = re.sub("\s+", " ",jsonString['Review'])
            if jsonString['ReviewerID'] in d:
                d[jsonString['ReviewerID']] = d[jsonString['ReviewerID']] + jsonString['Review']
            else:
                d[jsonString['ReviewerID']] = jsonString['Review']
                labels.append(jsonString['Feedback'])
                #d['label'] = jsonString['ReviewerID'] + "\t" + jsonString['Feedback']
    #print("Read {} documents".format(len(d.keys())))
    print("Read {} documents".format(len(d.values())))
    return list(d.values()), labels

def convert_text_into_features(X, stopwords_arg, analyzefn="word", range=(1,2)):
    # tokenizing, counting, and normaliziing 
    training_vectorizer = CountVectorizer(stop_words=stopwords_arg,
                                          analyzer=analyzefn,
                                          lowercase=True,
                                          ngram_range=range)
    X_features = training_vectorizer.fit_transform(X)
    return X_features, training_vectorizer

def convert_lines_to_feature_strings(lines, stopwords, remove_stopword_bigrams=True, apply_punc=True):

    print(" Converting from raw text to unigram and bigram features")
    if remove_stopword_bigrams:
        print(" Includes filtering stopword bigrams")
    if apply_punc:
        print(" Includes punctuation tokens")
        
    print(" Initializing")
    nlp          = English(parser=False)
    nlp.tokenizer.rules = {key: value for key, value in nlp.tokenizer.rules.items() if "'" not in key and "’" not in key and "‘" not in key}
    all_features = []
    print(" Iterating through documents extracting unigram and bigram features")
    for line in tqdm(lines):
        
        # Get spacy tokenization and normalize the tokens
        spacy_analysis    = nlp(line)
        spacy_tokens      = [token.orth_ for token in spacy_analysis]
        normalized_tokens = normalize_tokens(spacy_tokens)

        # Collect unigram tokens as features
        # Exclude unigrams that are stopwords or are punctuation strings (e.g. '.' or ',')
        if apply_punc:
            unigrams          = [token   for token in normalized_tokens
                                 if token not in stopwords]
        else:
            unigrams          = [token   for token in normalized_tokens
                                 if token not in stopwords and token not in string.punctuation]

        # Collect string bigram tokens as features
        bigrams = []
        bigram_tokens     = ["_".join(bigram) for bigram in bigrams]
        bigrams           = ngrams(normalized_tokens, 2) 
        if apply_punc:
            bigrams           = bigrams
        else:
            bigrams           = filter_punctuation_bigrams(bigrams)
        if remove_stopword_bigrams:
            bigrams = filter_stopword_bigrams(bigrams, stopwords)
        bigram_tokens = ["_".join(bigram) for bigram in bigrams]

        # Conjoin the feature lists and turn into a space-separated string of features.
        # E.g. if unigrams is ['coffee', 'cup'] and bigrams is ['coffee_cup', 'white_house']
        # then feature_string should be 'coffee cup coffee_cup white_house'
        feature_conjoin = unigrams+bigram_tokens 
        feature_string = []
        feature_string = ' '.join([str(elem) for elem in feature_conjoin])
        # Add this feature string to the output
        all_features.append(feature_string)


    #print(" Feature string for first document: '{}'".format(all_features[0]))
        
    return all_features

def convert_text_into_features_with_vocab(X, d, stopwords_arg, analyzefn="word", range=(1,2)):
    # tokenizing, counting, and normaliziing 
    training_vectorizer = CountVectorizer(stop_words=stopwords_arg,
                                          analyzer=analyzefn,
                                          lowercase=True,
                                          ngram_range=range)
    training_vectorizer.fit_transform(d)
    X_features = training_vectorizer.transform(X)
    return X_features, training_vectorizer

def whitespace_tokenizer(line):
    return line.split()

# Adapted from https://stackoverflow.com/questions/11116697/how-to-get-most-informative-features-for-scikit-learn-classifiers#11116960
def most_informative_features(vectorizer, classifier, n=20):
    # Adapted from https://stackoverflow.com/questions/11116697/how-to-get-most-informative-features-for-scikit-learn-classifiers#11116960
    feature_names       = vectorizer.get_feature_names()
    coefs_with_features = sorted(zip(classifier.coef_[0], feature_names))
    top                 = zip(coefs_with_features[:n], coefs_with_features[:-(n + 1):-1])
    for (coef_1, feature_1), (coef_2, feature_2) in top:
        print("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, feature_1, coef_2, feature_2))

def run_kfold(input_json_file, use_sklearn_feature_extraction, test_size, num_folds, stratify, random_seed):

    stop_words = load_stopwords(stopwords_file)
    X, y                              = read_and_clean_lines_classifier(input_json_file)
    X_train, X_test, y_train, y_test  = split_training_set(X, y, test_size)

    if use_sklearn_feature_extraction:
        X_features_train, training_vectorizer = convert_text_into_features(X_train, stop_words, "word", range=(1,2))
        X_test_documents = X_test
    else:
        print("Creating feature strings for training data")
        X_train_feature_strings = convert_lines_to_feature_strings(X_train, stop_words)
        X_features_train, training_vectorizer = convert_text_into_features(X_train_feature_strings, stop_words, whitespace_tokenizer)
    
    print("Doing cross-validation splitting with stratify={}. Showing 10 indexes for items in train/test splits in {} folds.".format(stratify,num_folds))
    if stratify:
        kfold = StratifiedKFold(n_splits=num_folds,shuffle=True,random_state=random_seed)
    else:
        kfold = KFold(n_splits=num_folds,shuffle=True,random_state=random_seed) 

    # Create the classifier object
    classifier = LogisticRegression(multi_class='multinomial', max_iter = 500)
    clf = svm.SVC()

    print("Running {}-fold cross-validation on {}% of the data, still holding out the rest for final testing.".format(num_folds,(1-test_size)*100))
    
    print("----------------------------Logistic Regression----------------------------")
    accuracy_scores = cross_val_score(classifier, X_features_train, y_train, scoring='accuracy', cv=kfold)
    print("accuracy scores = {}, mean = {}, stdev = {}".format(accuracy_scores, np.mean(accuracy_scores), np.std(accuracy_scores)))
    
    print("----------------------------Support Vector Classifier----------------------------")
    accuracy_scores = cross_val_score(clf, X_features_train, y_train, scoring='accuracy', cv=kfold)
    print("accuracy scores = {}, mean = {}, stdev = {}".format(accuracy_scores, np.mean(accuracy_scores), np.std(accuracy_scores)))

# Adapted from https://towardsdatascience.com/hyperparameter-tuning-c5619e7e6624
def examine_hyperparameters(X, Y):
    model = LogisticRegression(multi_class='multinomial', max_iter = 1000)
    model_svm = svm.SVC()

    #defining parameter range for LR
    solvers = ['newton-cg', 'lbfgs', 'sag', 'saga']
    penalty = ['l2']
    c_values = [10, 1, 0.1, 0.01]
    param_grid_lr = {'solvers': ['newton-cg', 'lbfgs', 'sag', 'saga'], 'penalty': ['l2'],'c_values': [10, 1, 0.1, 0.01]}
    
    param_grid_lr = 2
    #defining parameter range for SVM (adapted from the heatmap at https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html)
    param_grid_svm = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001],'kernel': ['rbf', 'poly']}

    # define grid search for LR
    grid = dict(solver=solvers,penalty=penalty,C=c_values)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
    grid_result_lr = grid_search.fit(X, Y)
    # summarize results for LR
    print("\n LR \nBest: %f using %s" % (grid_result_lr.best_score_, grid_result_lr.best_params_))
    means = grid_result_lr.cv_results_['mean_test_score']
    stds = grid_result_lr.cv_results_['std_test_score']
    params = grid_result_lr.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    # define grid search for SVM
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid_search = GridSearchCV(estimator=model_svm, param_grid=param_grid_svm, refit=True, cv=cv, scoring='accuracy', error_score=0)
    grid_result_svm = grid_search.fit(X, Y)
    # summarize results for SVM
    print("\n SVM \nBest: %f using %s" % (grid_result_svm.best_score_, grid_result_svm.best_params_))
    means = grid_result_svm.cv_results_['mean_test_score']
    stds = grid_result_svm.cv_results_['std_test_score']
    params = grid_result_svm.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    return grid_result_lr.best_params_, grid_result_svm.best_params_

def main(use_sklearn_feature_extraction, num_most_informative, plot_metrics):
    
    # Read in the stopword list
    print("\nLoading stopwords from {}".format(stopwords_file))
    stopwords = load_stopwords(stopwords_file)

    # Converting .csv file to .jsonl  
    print("\nConverting {} file to .jsonl".format(input_csv_reviews))
    csv_to_jsonl(input_csv_reviews, cutoff)
    
    # Reading and Extracting Informative data
    print("\n*******************Initial Data Analysis starts*****************")
    print("\nProcessing text from input file {}".format(input_json_reviews))

    # Running k-fold cross-validation (stratified by default because of the label imbalance)
    print("\nPerforming k-fold cross-validation (Stratified by default)")
    run_kfold(args.infile,args.use_sklearn_features,float(args.test_size),int(args.num_folds),args.no_stratify,int(args.seed))
    
    # Read the dataset in and split it into training documents/labels (X) and test documents/labels (y)
    X_train, X_test, y_train, y_test = split_training_set(*read_and_clean_lines(input_json_reviews))
    print("\ntraining example:\n"+X_train[0]+"\t"+y_train[0])
    print("\ntesting example:\n"+X_test[0]+"\t"+y_test[0])

    print("\nWriting reviews with positive feedback to {}".format(positive_reviews))
    write_user_reviews(X_train, y_train, positive_reviews, "positive")
    print("\nWriting reviews with moderate feedback to {}".format(moderate_reviews))
    write_user_reviews(X_train, y_train, moderate_reviews, "moderate")
    print("\nWriting reviews with negative feedback to {}".format(negative_reviews))
    write_user_reviews(X_train, y_train, negative_reviews, "negative")


    # PMI
    print("\nGetting unigram and bigram counts for positive reviews")
    with open(positive_reviews) as f:
        positive_reviews_text = f.readlines()
    positive_bigram_counts     = collect_bigram_counts(positive_reviews_text, stopwords, True, True)
    positive_unigram_w1_counts = get_unigram_counts(positive_bigram_counts,0)
    positive_unigram_w2_counts = get_unigram_counts(positive_bigram_counts,1)
    positive_reviews_pmi_score = compute_pmi(positive_bigram_counts, positive_unigram_w1_counts, positive_unigram_w2_counts) 
    print("\nTop positive reviews bigrams by frequency")
    print_sorted_items(positive_bigram_counts, topN_to_show, 'descending')
    print("\nTop positive bigrams by PMI scores")
    print_sorted_items(positive_reviews_pmi_score, topN_to_show, 'descending')

    print("\nGetting unigram and bigram counts for moderate reviews")
    with open(moderate_reviews) as f:
        moderate_reviews_text = f.readlines()
    moderate_bigram_counts     = collect_bigram_counts(moderate_reviews_text, stopwords, True, True)
    moderate_unigram_w1_counts = get_unigram_counts(moderate_bigram_counts,0)
    moderate_unigram_w2_counts = get_unigram_counts(moderate_bigram_counts,1)
    moderate_reviews_pmi_score = compute_pmi(moderate_bigram_counts, moderate_unigram_w1_counts, moderate_unigram_w2_counts) 
    print("\nTop moderate reviews bigrams by frequency")
    print_sorted_items(moderate_bigram_counts, topN_to_show, 'descending')
    print("\nTop moderate bigrams by PMI scores")
    print_sorted_items(moderate_reviews_pmi_score, topN_to_show, 'descending')
    
    print("\nGetting unigram and bigram counts for negative reviews")
    with open(negative_reviews) as f:
        negative_reviews_text = f.readlines()
    negative_bigram_counts     = collect_bigram_counts(negative_reviews_text, stopwords, True, True)
    negative_unigram_w1_counts = get_unigram_counts(negative_bigram_counts,0)
    negative_unigram_w2_counts = get_unigram_counts(negative_bigram_counts,1)
    negative_pmi_score         = compute_pmi(negative_bigram_counts, negative_unigram_w1_counts, negative_unigram_w2_counts) 
    print("\nTop negative reviews bigrams by frequency")
    print_sorted_items(negative_bigram_counts, topN_to_show, 'descending')
    print("\nTop negative bigrams by PMI scores")
    print_sorted_items(negative_pmi_score, topN_to_show, 'descending')
  

    print("Computing LLR scores by frequency")
    total_positive_counts = positive_bigram_counts + positive_unigram_w1_counts + positive_unigram_w2_counts
    total_moderate_counts = moderate_bigram_counts + moderate_unigram_w1_counts + moderate_unigram_w2_counts
    total_negative_counts = negative_bigram_counts + negative_unigram_w1_counts + negative_unigram_w2_counts
    positive_llr_score         = compute_llr(total_positive_counts, total_negative_counts+total_moderate_counts)
    ranked = sorted(positive_llr_score.items(), key=lambda x: x[1])

    d = dict()
  #  print("\nMore in Moderate and Negative Reviews")
    for k,v in ranked[:topN_to_show]:
        k = k.replace('_',' ')
        d[k] = k
    
  #  print("\nMore in Positive Reviews")
    for k,v in ranked[-topN_to_show:]:
        k = k.replace('_',' ')
        d[k] = k

    print("\n*************Initial Data Analysis ends*************")

    # Building Baseline Binary Classifier
    print("\n***************Building Classifier starts*****************")

    # list of unique users
    users = get_users(input_json_reviews)

    # Building Classifier starts
    print("\nProcessing text from input file {}".format(input_json_reviews))
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = split_training_set(*read_and_clean_lines_classifier(input_json_reviews))
    
    #lines, labels = read_and_clean_lines_classifier(input_json_file)

    print("\nWriting positive reviews to {}".format(positive_reviews))
    X_train_lines_pos = write_user_reviews_clf(X_train_clf, y_train_clf, positive_reviews, "positive")
    
    print("\nWriting moderate reviews to {}".format(moderate_reviews))
    X_train_lines_mod = write_user_reviews_clf(X_train_clf, y_train_clf, moderate_reviews, "moderate")
    
    print("\nWriting negative reviews to {}".format(negative_reviews))
    X_train_lines_neg = write_user_reviews_clf(X_train_clf, y_train_clf, negative_reviews, "negative")
    
    print("\nCreating feature strings for training data")
    X_train_feature_strings = convert_lines_to_feature_strings(X_train_clf, stopwords, True, False)
    print("\nCreating feature strings for test data")
    X_test_feature_strings  = convert_lines_to_feature_strings(X_test_clf,  stopwords, True, False)

    X_features_train, training_vectorizer = convert_text_into_features(X_train_feature_strings, stopwords, whitespace_tokenizer)
    X_features_train_vocab, training_vectorizer_vocab = convert_text_into_features_with_vocab(X_train_feature_strings, d, stopwords, whitespace_tokenizer)

    # Create a logistic regression classifier trained on the featurized training data
    lr_classifier = LogisticRegression(multi_class='multinomial', max_iter = 1000)
    lr_classifier_improved = LogisticRegression(multi_class='multinomial', max_iter = 1000)
    lr_classifier.fit(X_features_train, y_train_clf)
    lr_classifier_improved.fit(X_features_train_vocab, y_train_clf)

    # Show which features have the highest-value logistic regression coefficients
    print("\nMost informative features with the baseline LR classifier")
    most_informative_features(training_vectorizer, lr_classifier, num_most_informative)

    print("\nMost informative features with the LR classifier using vocab generated from LLR")
    most_informative_features(training_vectorizer_vocab, lr_classifier_improved, num_most_informative)

    # Apply the "vectorizer" created using the training data to the test documents, to create testset feature vectors
    X_test_features =  training_vectorizer.transform(X_test_feature_strings)
    X_test_features_vocab =  training_vectorizer_vocab.transform(X_test_feature_strings)
    
    print("\nClassifying test data using baseline LR")
    predicted_labels = lr_classifier.predict(X_test_features)
    print('Accuracy  = {}'.format(metrics.accuracy_score(predicted_labels,  y_test_clf)))
    for label in ['positive', 'moderate', 'negative']:
        print('Precision for label {} = {}'.format(label, metrics.precision_score(predicted_labels, y_test_clf, pos_label=label, average='weighted')))
        print('Recall    for label {} = {}'.format(label, metrics.recall_score(predicted_labels,    y_test_clf, pos_label=label, average='weighted')))
    
    if plot_metrics:
        print("\nGenerating confusion matrix for baseline LR")
        metrics.plot_confusion_matrix(lr_classifier, X_test_features, y_test_clf, normalize='true')
        #metrics.plot_roc_curve(lr_classifier, X_test_features, y_test_clf)
        plt.title("Baseline LR")
        plt.show()
    
    print("\nClassifying test data using improvised LR (w/ llr vocab)")
    predicted_labels_improved = lr_classifier_improved.predict(X_test_features_vocab)
    print('Accuracy  = {}'.format(metrics.accuracy_score(predicted_labels_improved,  y_test_clf)))
    for label in ['positive', 'moderate', 'negative']:
        print('Precision for label {} = {}'.format(label, metrics.precision_score(predicted_labels_improved, y_test_clf, pos_label=label, average='weighted')))
        print('Recall    for label {} = {}'.format(label, metrics.recall_score(predicted_labels_improved, y_test_clf, pos_label=label, average='weighted')))
    
    if plot_metrics:
        print("\nGenerating confusion matrix for improvised LR (w/ llr vocab)")
        metrics.plot_confusion_matrix(lr_classifier_improved, X_test_features_vocab, y_test_clf, normalize='true')
        #metrics.plot_roc_curve(lr_classifier_improved, X_test_features_vocab, y_test_clf)
        plt.title("Improvised LR (w/ llr vocab)")
        plt.show()

    
    # Implementing baseline SVM
    svm_classifier = svm.SVC()
    svm_classifier.fit(X_features_train, y_train_clf)
    
    # Implementing baseline SVM with added Vocabulary features
    svm_classifier_vocab = svm.SVC()
    svm_classifier_vocab.fit(X_features_train_vocab, y_train_clf)

    # Show which features have the highest-value SVM coefficients
    #print("\nMost informative features with the baseline SVM classifier")
    #most_informative_features(training_vectorizer, svm_classifier, num_most_informative)

    print("\nClassifying test data with baseline SVM")
    predicted_labels = svm_classifier.predict(X_test_features)
    print('Accuracy  = {}'.format(metrics.accuracy_score(predicted_labels,  y_test_clf)))
    for label in ['positive', 'moderate', 'negative']:
        print('Precision for label {} = {}'.format(label, metrics.precision_score(predicted_labels, y_test_clf, pos_label=label, average='weighted')))
        print('Recall    for label {} = {}'.format(label, metrics.recall_score(predicted_labels,    y_test_clf, pos_label=label, average='weighted')))
    
    if plot_metrics:
        print("\nGenerating confusion matrix for baseline SVM")
        metrics.plot_confusion_matrix(svm_classifier, X_test_features, y_test_clf, normalize='true')
  #      metrics.plot_roc_curve(svm_classifier, X_test_features, y_test_clf)
        plt.title("baseline SVM")
        plt.show()
    
    print("\nClassifying test data with improvised SVM (w/ llr vocab)")
    predicted_labels_vocab = svm_classifier_vocab.predict(X_test_features_vocab)
    print('Accuracy  = {}'.format(metrics.accuracy_score(predicted_labels_vocab,  y_test_clf)))
    for label in ['positive', 'moderate', 'negative']:
        print('Precision for label {} = {}'.format(label, metrics.precision_score(predicted_labels_vocab, y_test_clf, pos_label=label, average='weighted')))
        print('Recall    for label {} = {}'.format(label, metrics.recall_score(predicted_labels_vocab,    y_test_clf, pos_label=label, average='weighted')))
    
    if plot_metrics:
        print("\nGenerating confusion matrix for improvised SVM (w/ llr vocab)")
        metrics.plot_confusion_matrix(svm_classifier_vocab, X_test_features_vocab, y_test_clf, normalize='true')
   #     metrics.plot_roc_curve(svm_classifier_vocab, X_test_features_vocab, y_test_clf)
        plt.title("Improvised SVM (w/ llr vocab)")    
        plt.show()
        
    print("\n*****************Building Classifier ends**************")
    # Building Classifier ends

    # hyper-parameter tuning
    #print("\nSkipping Hyperparameter tuning...")
    print("\nHyperparameter tuning... (might take a couple of mins)")
    print("\nTuning baseline LR and SVM models... (might take a couple of mins)")
    best_params_LR, best_params_SVM = examine_hyperparameters(X_features_train, y_train_clf)
    print("\nTuning improvised LR and SVM models (w/ llr vocab)... (might take a couple of mins)")
    best_params_LR_with_vocab, best_params_SVM_with_vocab = examine_hyperparameters(X_features_train_vocab, y_train_clf)
    
    # Logistic Regression after hyperparameter tuning
    print("\nClassifying with LR 2.0 using the best parameters")
    
    # Create a logistic regression classifier trained on the featurized training data
    lr_classifier2 = LogisticRegression(multi_class='multinomial', penalty=best_params_LR['penalty'],C=best_params_LR['C'],solver=best_params_LR['solver'], max_iter = 500)
    lr_classifier2_with_vocab = LogisticRegression(multi_class='multinomial', penalty=best_params_LR_with_vocab['penalty'], C=best_params_LR_with_vocab['C'], solver=best_params_LR_with_vocab['solver'], max_iter = 1000)
    lr_classifier2.fit(X_features_train, y_train_clf)
    lr_classifier2_with_vocab.fit(X_features_train_vocab, y_train_clf)

    # Show which features have the highest-value logistic regression coefficients
    print("\nMost informative features with baseline LR 2.0 classifier")
    most_informative_features(training_vectorizer, lr_classifier2, num_most_informative)
    
    print("\nMost informative features with improvised LR 2.0 classifier (w/ custom llr vocabulary)")
    most_informative_features(training_vectorizer_vocab, lr_classifier2_with_vocab, num_most_informative)

    print("\nClassifying test data with LR 2.0")
    predicted_labels = lr_classifier2.predict(X_test_features)
    print('Accuracy  = {}'.format(metrics.accuracy_score(predicted_labels,  y_test_clf)))
    for label in ['positive', 'moderate', 'negative']:
        print('Precision for label {} = {}'.format(label, metrics.precision_score(predicted_labels, y_test_clf, pos_label=label, average='weighted')))
        print('Recall    for label {} = {}'.format(label, metrics.recall_score(predicted_labels,    y_test_clf, pos_label=label, average='weighted')))
 
    if plot_metrics:
        print("\nGenerating confusion matrix for baseline LR 2.0 classifier")
        metrics.plot_confusion_matrix(lr_classifier2, X_test_features, y_test_clf, normalize='true')
    #   metrics.plot_roc_curve(lr_classifier2, X_test_features, y_test_clf)
        plt.title("Baseline LR 2.0 classifier")
        plt.show()

    print("\nClassifying test data with improvised LR 2.0 (w/ llr vocab)")
    predicted_labels_with_vocab = lr_classifier2_with_vocab.predict(X_test_features_vocab)
    print('Accuracy  = {}'.format(metrics.accuracy_score(predicted_labels_with_vocab,  y_test_clf)))
    for label in ['positive', 'moderate', 'negative']:
        print('Precision for label {} = {}'.format(label, metrics.precision_score(predicted_labels_with_vocab, y_test_clf, pos_label=label, average='weighted')))
        print('Recall    for label {} = {}'.format(label, metrics.recall_score(predicted_labels_with_vocab,    y_test_clf, pos_label=label, average='weighted')))
    
    if plot_metrics:
        print("\nGenerating confusion matrix for improvised LR 2.0 classifier (w/ llr vocab)")
        metrics.plot_confusion_matrix(lr_classifier2_with_vocab, X_test_features_vocab, y_test_clf, normalize='true')
    #   metrics.plot_roc_curve(lr_classifier2_with_vocab, X_test_features_vocab, y_test_clf)
        plt.title("LR 2.0 classifier (w/ llr vocab)")
        plt.show()

        
    # SVM after hyperparameter tuning
    print("\nClassifying w/ SVM 2.0 using the best parameters")
    # Create a svm classifier trained on the featurized training data
    svm_classifier2 = svm.SVC(gamma=best_params_SVM['gamma'],C=best_params_SVM['C'],kernel=best_params_SVM['kernel'])
    svm_classifier2.fit(X_features_train, y_train_clf)

    svm_classifier2_vocab = svm.SVC(gamma=best_params_SVM_with_vocab['gamma'],C=best_params_SVM_with_vocab['C'],kernel=best_params_SVM_with_vocab['kernel'])
    svm_classifier2_vocab.fit(X_features_train_vocab, y_train_clf)


    # Show which features have the highest-value logistic regression coefficients
    #print("\nMost informative features with the improved SVM classifier")
    #most_informative_features(training_vectorizer, svm_classifier2, num_most_informative)
    
    print("\nClassifying test data with SVM 2.0")
    predicted_labels = svm_classifier2.predict(X_test_features)
    print('Accuracy  = {}'.format(metrics.accuracy_score(predicted_labels,  y_test_clf)))
    for label in ['positive', 'moderate', 'negative']:
        print('Precision for label {} = {}'.format(label, metrics.precision_score(predicted_labels, y_test_clf, pos_label=label, average='weighted')))
        print('Recall    for label {} = {}'.format(label, metrics.recall_score(predicted_labels,    y_test_clf, pos_label=label, average='weighted')))
    
    if plot_metrics:
        print("\nGenerating confusion matrix for baseline SVM 2.0")
        metrics.plot_confusion_matrix(svm_classifier2, X_test_features, y_test_clf, normalize='true')
    #    metrics.plot_roc_curve(svm_classifier2, X_test_features, y_test_clf)
        plt.title("Baseline SVM 2.0")
        plt.show()

    print("\nClassifying test data with improvised SVM 2.0 (w/ llr vocab)")
    predicted_labels_vocab = svm_classifier2_vocab.predict(X_test_features_vocab)
    print('Accuracy  = {}'.format(metrics.accuracy_score(predicted_labels_vocab,  y_test_clf)))
    for label in ['positive', 'moderate', 'negative']:
        print('Precision for label {} = {}'.format(label, metrics.precision_score(predicted_labels, y_test_clf, pos_label=label, average='weighted')))
        print('Recall    for label {} = {}'.format(label, metrics.recall_score(predicted_labels,    y_test_clf, pos_label=label, average='weighted')))
    
    if plot_metrics:
        print("\nGenerating confusion matrix for improvised SVM 2.0 (w/ llr vocab)")
        metrics.plot_confusion_matrix(svm_classifier2_vocab, X_test_features_vocab, y_test_clf, normalize='true')
    #    metrics.plot_roc_curve(svm_classifier2_vocab, X_test_features_vocab, y_test_clf)
        plt.title("SVM 2.0 (w/ llr vocab)")
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Options for running this script')
    parser.add_argument('--use_sklearn_features', default=False, action='store_true', help="Use sklearn's feature extraction")
    parser.add_argument('--test_size',            default=0.3,   action='store',      help="Proportion (from 0 to 1) of items held out for final testing")
    parser.add_argument('--num_most_informative', default=10, action='store', help="Number of most-informative features to show")
    parser.add_argument('--num_folds',            default=5,     action='store',      help="Number of folds for cross-validation (use 2 for just a train/test split)")
    parser.add_argument('--no_stratify',             default=True, action='store_false', help="Use non-stratified cross-validation")
    parser.add_argument('--seed',                 default=13,    action='store',      help="Random seed")
    parser.add_argument('--plot_metrics', default=False, action='store_true', help="Generate figures for evaluation")
    parser.add_argument('--infile',               default="data/Kindle_Store_5.jsonl",  action='store',      help="Input jsonlines file")
    args = parser.parse_args()
    main(args.use_sklearn_features, int(args.num_most_informative), args.plot_metrics)
