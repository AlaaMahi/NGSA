#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd

def main(tuning_number=2):

	df = pd.read_csv("data/data_features_labels.csv")

	best_features = ["in_degree_target", "out_degree_target", "title_overlap", 
	                 "temp_diff", "abstract_overlap", "jaccard_coefficient", 
	                 "adamic_adar", "pref_attachment", "common_neighbors", 
	                 "abstract_cosine_similarity"]

	X = df[best_features]
	y = df.label

	# param_test = {"n_estimators": range(20, 81, 10)}
	param_test = {"max_depth": range(3, 10, 2), "min_samples_split": range(200, 1001, 400)}
	# param_test = {"min_samples_leaf": range(30, 71, 10)}

	gsearch = GridSearchCV(estimator = GradientBoostingClassifier(n_estimators=50),
	                        param_grid=param_test, scoring="f1", n_jobs=4, iid=False, cv=5,
	                        verbose=2)

	gsearch.fit(X, y)
	                        
	with open("gradient_boosting_hyperparameters_{}.txt".format(tuning_number), "w") as f:
		for score in gsearch.grid_scores_:
			f.write(str(score))
			f.write("\n")

if __name__ == "__main__":
	main()