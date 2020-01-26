#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This class creates features from the initial data.
"""
import networkx as nx
import pandas as pd
import copy as cp
import time
import csv
import os

from utils import compare_texts, count_common_authors, jaccard_coefficient,\
                  adamic_adar, pref_attachment, katz_centrality, common_neighbors,\
                  Word2vec, BoV, nodes_to_edge_embedding

class FeatureEngineering(object):

    def __init__(self, df_node_info, node_embeddings=None, path_word_embeddings=None, data="test"):
        """        
        """
        # Open the network as a DataFrame (train or test)
        print("--> Opening the source-node data")

        with open("data/{}ing_set.txt".format(data), "r") as f:
            reader = csv.reader(f)
            data_list = list(reader)

        columns = ["source", "target"]
        if data == "train":
            columns.append("label")

        data_list = [element[0].split(" ") for element in data_list]
        self.df = pd.DataFrame(data_list, columns=columns)

        if data == "train":
            self.df["label"] = self.df.label.map(pd.to_numeric)

        # Add the edge embeddings
        print("--> Computing and adding the edge embeddings")
        if node_embeddings is not None:
            embedding_dim = next(iter(node_embeddings.values())).shape[0]

            self.embedding_columns = ["emb{}".format(i+1) for i in range(embedding_dim)]

            self.df[self.embedding_columns] = self.df.apply(lambda row: pd.Series(
                nodes_to_edge_embedding(row["source"], row["target"], node_embeddings)), axis=1)

        # Add the node information
        print("--> Merging the source-node data with the node information")

        df_node_info_source = cp.deepcopy(df_node_info)
        df_node_info_source.columns = [col + "_source" 
                                       for col in df_node_info.columns]
        self.df = self.df.join(df_node_info_source, on="source")

        df_node_info_target = cp.deepcopy(df_node_info)
        df_node_info_target.columns = [col + "_target" 
                                       for col in df_node_info.columns]
        self.df = self.df.join(df_node_info_target, on="target")
        
        # Use the network of the training set for creating some features
        # (e.g Adamic Adar)
        if data == "test":
            with open("data/training_set.txt".format(data), "r") as f:
                reader = csv.reader(f)
                data_list = list(reader)

            columns.append("label")
            data_list = [element[0].split(" ") for element in data_list]
            self.df_train = pd.DataFrame(data_list, columns=columns)
            self.df_train["label"] = self.df_train.label.map(pd.to_numeric)

        # Data for creating the graph
        if data == "train":
            data_for_nx = cp.deepcopy(self.df)
        elif data == "test":
            data_for_nx = cp.deepcopy(self.df_train)

        # Create an undirected graph
        print("--> Creating an undirected graph from the training data")
        self.G = nx.Graph()

        # Add the nodes from the source and target columns
        self.G.add_nodes_from(data_for_nx["source"])
        self.G.add_nodes_from(data_for_nx["target"])

        # Add the edges
        edges = [(data_for_nx["source"][i], data_for_nx["target"][i]) for i in range(len(data_for_nx))
                      if data_for_nx["label"][i] == 1]
        self.G.add_edges_from(edges)

        # Create a directed graph
        print("--> Creating a directed graph from the training data")
        self.G_dir = nx.DiGraph()

        # Add the nodes from the source and target columns
        self.G_dir.add_nodes_from(data_for_nx["source"])
        self.G_dir.add_nodes_from(data_for_nx["target"])

        # Add the edges
        edges = [(data_for_nx["source"][i], data_for_nx["target"][i]) for i in range(len(data_for_nx))
                if data_for_nx["label"][i] == 1]
        self.G_dir.add_edges_from(edges)

        # Free up memory
        del data_for_nx

        # Path to word embeddings
        self.path_word_embeddings = path_word_embeddings

        # Available features that we can create
        self.available_features = [func for func in dir(self) 
                                   if not func.startswith("__") 
                                   and hasattr(getattr(self, func), "__call__")
                                   and func != "create_features"]

    def create_features(self, features=[]):
        """Create specific features.    
        """     
        for feature_name in features:

            print("--> Creating the feature '{}' ...".format(feature_name))
            start = time.time()

            try:
                getattr(self, feature_name)()
            except AttributeError:
                raise ValueError("'{}' is not among the available features that "
                                 "can be created: {}".format(feature_name, 
                                    self.available_features))

            end = time.time()
            time_elapsed = end - start

            print("it took {}min {}s".format(round(time_elapsed//60), 
                                            round(time_elapsed%60)))

    def title_overlap(self):
        """Common words in the title between the source and the target nodes.
        """
        self.df["title_overlap"] = self.df.apply(lambda row: 
                                       compare_texts(row["title_processed_source"], 
                                                     row["title_processed_target"]), 
                                       axis=1)

    def temp_diff(self):
        """Year difference between the source and the target nodes.
        """
        self.df["temp_diff"] = self.df.apply(lambda row: 
                                             row["year_source"]-row["year_target"], 
                                             axis=1)

    def abstract_overlap(self):
        """Common words in the abstract between the source and the target nodes.
        """
        self.df["abstract_overlap"] = self.df.apply(lambda row: 
                                                    compare_texts(row["abstract_processed_source"], 
                                                        row["abstract_processed_target"]),
                                                    axis=1)
    def is_same_journal(self):
        """Do the source and target node belong to the same journal ?
        """
        self.df["is_same_journal"] = ((self.df.journal_name_source == self.df.journal_name_target) 
                                     & (self.df.journal_name_source != "")).astype("int")

    def jaccard_coefficient(self):
        self.df["jaccard_coefficient"] = self.df.apply(lambda row: 
            jaccard_coefficient(row["neighbors_source"], row["neighbors_target"]), 
            axis=1)

    def common_neighbors(self):
        self.df["common_neighbors"] = self.df.apply(lambda row: 
            common_neighbors(row["neighbors_source"], row["neighbors_target"]), 
            axis=1)

    def adamic_adar(self):
        preds_dico = adamic_adar(self.df, self.G)

        self.df["adamic_adar"] = self.df.apply(lambda row: 
            preds_dico[(row["source"], row["target"])], 
            axis=1)

    def pref_attachment(self):
        preds_pa_dico = pref_attachment(self.df, self.G)

        self.df["pref_attachment"] = self.df.apply(lambda row: 
            preds_pa_dico[(row["source"], row["target"])], 
            axis=1)

    def katz_centrality(self):
        centrality_dico = katz_centrality(self.G)

        self.df["katz_centrality_source"] = self.df.apply(lambda row: 
            centrality_dico[row["source"]], 
            axis=1)

        self.df["katz_centrality_target"] = self.df.apply(lambda row: 
            centrality_dico[row["target"]], 
            axis=1)

    def has_path(self):
        """Takes approximately 12 minutes on the training set.
        """
        self.df["has_path"] = self.df.apply(lambda row: 
            int(nx.has_path(self.G_dir, row["source"], row["target"])),
            axis=1)

    def shortest_path(self):
        """Takes approximately 10 minutes on the training set.
        """
        self.df["shortest_path"] = self.df.apply(lambda row: 
            nx.shortest_path_length(self.G_dir, source=row["source"], target=row["target"])
            if row["has_path"] == 1
            else -1,
            axis=1)

    def max_flow(self):
        self.df["max_flow"] = self.df.apply(lambda row: 
            nx.maximum_flow_value(self.G, row["source"], row["target"]),
            axis=1)

    def in_degree_source(self):
        self.df["in_degree_source"] = self.df.apply(lambda row: 
            self.G_dir.in_degree(row["source"]),
            axis=1)

    def in_degree_target(self):
        self.df["in_degree_target"] = self.df.apply(lambda row: 
            self.G_dir.in_degree(row["target"]),
            axis=1)

    def out_degree_source(self):
        self.df["out_degree_source"] = self.df.apply(lambda row: 
            self.G_dir.out_degree(row["source"]),
            axis=1)

    def out_degree_target(self):
        self.df["out_degree_target"] = self.df.apply(lambda row: 
            self.G_dir.out_degree(row["target"]),
            axis=1)

    def abstract_cosine_similarity(self):
        # Load pretrained word embeddings
        w2v = Word2vec(os.path.join(self.path_word_embeddings, "crawl-300d-200k.vec"), 
                       nmax=200000)
        s2v = BoV(w2v)

        self.df["abstract_cosine_similarity"] = self.df.apply(lambda row: 
            s2v.score(row["abstract_embedding_source"], row["abstract_embedding_target"]), 
            axis=1)

 