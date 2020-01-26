#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This scripts contains useful functions used in this project.
"""
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import nltk
import re 
import io

from sklearn.ensemble import RandomForestClassifier

# English stopwords
STOPWORDS = set(nltk.corpus.stopwords.words("english"))

#### Functions used for processing the node information ####

def format_author_name(author_name):
    """Format the author name (string) as N Benabderrazik. If there are 
    multiple first names it would be: A B FamilyName where A and B are 
    the first letters of the respective first names.
    """
    # Keep only strings before an opening parenthesis (e.g. discard (Montpellier))
    author_name = re.sub(r"\(.*$", "", author_name)
    
    # Keep only word characters
    author_name = re.findall(r"[\w']+", author_name)
    
    # Turn the first names to initials and keep the family name as is
    author_name = [author_name[i][0] if i < len(author_name)-1 else author_name[i] 
              for i in range(len(author_name))]
    
    return " ".join(author_name)

def format_author_names(author_names):
    """Format a list of author names.
    """
    return [format_author_name(author_name) for author_name in author_names]

def process_text(t):
    """Remove stopwords, lower and tokenize
    """
    result = t.lower().split(" ")
    result = [token for token in result if token not in STOPWORDS]

    return result

def nodes_to_edge_embedding(source_id, target_id, node_embeddings):
    """Get the embedding for an edge using the hadamard product between the 
    nodes embeddings
    
    Parameters
    ----------
    source_id : str
        Id of the source node.
    target_id : str
        Id of the target node.
    node_embeddings : dict
        Dictionary mapping a node id (str) to its d dimensional node2vec 
        embedding.
    """
    # Embedding dimension
    embedding_dim = next(iter(node_embeddings.values())).shape[0]

    # Source and target nodes embeddings
    source_node_embedding = node_embeddings[source_id] if source_id in node_embeddings else np.zeros(embedding_dim)
    target_node_embedding = node_embeddings[target_id] if target_id in node_embeddings else np.zeros(embedding_dim)

    # Hadamard product
    edge_embedding = np.multiply(source_node_embedding, target_node_embedding)

    return edge_embedding

class Word2vec():
    def __init__(self, fname, nmax=100000):
        self.load_wordvec(fname, nmax)
        self.word2id = dict(zip(self.word2vec.keys(), range(len(self.word2vec.keys()))))
        self.id2word = {v: k for k, v in self.word2id.items()}
        #self.embeddings = np.array(self.word2vec.values())
    
    def load_wordvec(self, fname, nmax):
        self.word2vec = {}
        with io.open(fname, encoding='utf-8') as f:
            next(f)
            for i, line in enumerate(f):
                word, vec = line.split(' ', 1)
                self.word2vec[word] = np.fromstring(vec, sep=' ')
                if i == (nmax - 1):
                    break
        print('Loaded %s pretrained word vectors' % (len(self.word2vec)))

    def score(self, w1, w2):
        # cosine similarity: np.dot  -  np.linalg.norm
        w1_embd = self.word2vec[w1]
        w2_embd = self.word2vec[w2]
        
        cosine_sim = w1_embd.dot(w2_embd)/(np.linalg.norm(w1_embd)*np.linalg.norm(w2_embd))
        return cosine_sim

class BoV():
    def __init__(self, w2v):
        self.w2v = w2v
    
    def encode(self, sentences, idf=False):
        # takes a list of sentences, outputs a numpy array of sentence embeddings
        # see TP1 for help
        sentemb = []
        for sent in sentences:
            if idf is False:
                # mean of word vectors
                # number of words for which we have an embedding
                n_words = sum([1 for word in sent if word in self.w2v.word2id])
                
                # compute sentence embedding only if we have the embedding of at least one word
                if n_words > 0:
                    # vector with the sum of all word embeddings element-wise
                    sum_embd = np.sum(np.array([self.w2v.word2vec[word] for word in sent if word in self.w2v.word2id]), axis=0)
                    mean_embd = sum_embd/n_words
                    sentemb.append(np.array(mean_embd))
                # if no embedding found, append a random embedding
                else:
                    sentemb.append(np.random.rand(300))
            else:
                # idf-weighted mean of word vectors
                assert False, 'TODO: fill in the blank'
        return np.vstack(sentemb)

    def most_similar(self, s, sentences, idf=False, K=5):
        # get most similar sentences and **print** them
        keys = self.encode(sentences, idf)
        query = self.encode([s], idf)[0]
        
        scores = []
        for key in keys:
            score = self.score(query, key)
            scores.append(score)
        scores = np.array(scores)
        argsorted_scores = np.argsort(scores)[::-1]
        return [" ".join(sentences[arg]) for arg in argsorted_scores[1:K+1]]

    def score(self, s1, s2, idf=False):
        # cosine similarity: use   np.dot  and  np.linalg.norm
        cosine_sim = s1.dot(s2)/(np.linalg.norm(s1)*np.linalg.norm(s2))
        return cosine_sim

#### Functions used to compute compute features ####

def compare_texts(t1, t2):
    n_overlapping_words = sum([1 for token in t1 if token in t2])
    return n_overlapping_words

def count_common_authors(l_authors1, l_authors2):
    n_common_authors = sum([1 for author in l_authors1 if author in l_authors2])
    return n_common_authors

def jaccard_coefficient(neighbors_target, neighbors_source):
    """Compute the jaddard coefficient between a source and a target node. 
    The neighbors are represented by a list of strings.
    """
    common_neighbors = len(set(neighbors_source).intersection(set(neighbors_target)))
    total_neighbors = len(neighbors_source+neighbors_target)

    return common_neighbors/total_neighbors

def common_neighbors(neighbors_target, neighbors_source):
    """Compute the common neighbors between a source and a target node. 
    """
    common_neighbors = len(set(neighbors_source).intersection(set(neighbors_target)))

    return common_neighbors

def adamic_adar(df, G):
    to_compute = [(df['source'][i], df['target'][i]) for i in range(len(df))] 
    preds = nx.adamic_adar_index(G, to_compute)
    
    preds_dico = {(source, target): sim for (source, target, sim) in preds}
    
    return preds_dico

def pref_attachment(df, G):
    to_compute = [(df['source'][i], df['target'][i]) for i in range(len(df))]
    preds_pa = nx.preferential_attachment(G, to_compute)
    
    preds_pa_dico = {(source, target): sim for (source, target, sim) in preds_pa}
    
    return preds_pa_dico

def katz_centrality(G):
    centrality_dico = nx.katz_centrality(G)
    
    return centrality_dico

#### Functions for making predictions ####

def complete_predictions(X_test, predictions_positive_temp_diff):
    """Make predictions on the test set. Articles can't cite articles in the 
    future. 
    
    Parameters
    ----------
    X_test : DataFrame
        Test data with its features.
    predictions_positive_temp_diff : list
        Predictions for source and target nodes with temp_diff >= 0.
    
    Returns
    -------
    list
    """
    predictions = []
    idx_pred = 0
    for row in X_test.iterrows():
        if row[1]["temp_diff"] < 0:
            predictions.append(0)
        else:
            predictions.append(predictions_positive_temp_diff[idx_pred])
            idx_pred +=1
    return predictions

#### Other useful functions ####

def random_forest_features_importances(X, y, n_estimators=10):

    # Initialize Random Forest classifier
    clf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, verbose=1)

    # Features used 
    features = list(X.columns)

    # Fit the classifier
    print("Training ...")
    clf.fit(X, y)
    importances = clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. feature %s (%f)" % (f + 1, features[indices[f]], 
                                       importances[indices[f]]))

    # Top features
    top_features = [features[index] for index in indices]

    # Plot the feature importances of the clf
    plt.figure(figsize=(16, 9))
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), top_features, 
               rotation=45)
    plt.xlim([-1, X.shape[1]])
    plt.show()

    return top_features