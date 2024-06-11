("***KNN***")
#Author:Joelle Schiffmann

#Import Liberaries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, f_classif

# Import from "Preprocessing.py"
from Preprocessing import *
#Import the evaulation function
from Model_evaluation import *

# Feature Selection for balanced & unbalanced data with SelectKBest
kbest_selector = SelectKBest(score_func=f_classif, k=10)

# For unbalanced data
kbest_selector.fit(X_train, y_train)
X_train_knn_unbalanced = kbest_selector.transform(X_train)
X_test_knn_unbalanced = kbest_selector.transform(X_test)
X_valid_knn_unbalanced = kbest_selector.transform(X_valid)
selected_features_unbalanced_knn = X_train.columns[kbest_selector.get_support()]
print("Remaining features after SelectKBest feature selection with unbalanced data for KNN:")
print(selected_features_unbalanced_knn)
#10 remained:'baseline_value', 'accelerations', 'prolonged_decelerations', 'pc_short_term_ab_variability', 'mean_short_term_variability', 'pc_long_term_ab_variability', 'histogram_mode', 'histogram_mean', 'histogram_median', 'histogram_variance'

# For balanced data
kbest_selector.fit(X_train_balanced, y_train_balanced)
X_train_knn_balanced = kbest_selector.transform(X_train_balanced)
X_test_knn_balanced = kbest_selector.transform(X_test)
X_valid_knn_balanced = kbest_selector.transform(X_valid)
selected_features_balanced_knn = X_train.columns[kbest_selector.get_support()]
print("Remaining features after SelectKBest feature selection with balanced data for KNN:")
print(selected_features_balanced_knn)
#10 remained: 'baseline_value', 'accelerations', 'prolonged_decelerations', 'pc_short_term_ab_variability', 'mean_short_term_variability', 'mean_long_term_variability', 'histogram_mode', 'histogram_mean', 'histogram_median', 'histogram_variance'],

# K-Nearest Neighbors with balanced data and feature selection
name = "K-Nearest Neighbors with balanced data and feature selection"
knn_balanced_FS = KNeighborsClassifier(n_neighbors=10)
knn_balanced_FS.fit(X_train_knn_balanced, y_train_balanced)
evaluate_model_performance(knn_balanced_FS, X_test_knn_balanced, y_test, name)

# K-Nearest Neighbors with unbalanced data and feature selection
name = "K-Nearest Neighbors with unbalanced data and feature selection"
knn_unbalanced_FS = KNeighborsClassifier(n_neighbors=10)
knn_unbalanced_FS.fit(X_train_knn_unbalanced, y_train)
evaluate_model_performance(knn_unbalanced_FS, X_test_knn_unbalanced, y_test, name)
#best Performance in all metrics: Accuracy: 0.9481, Precision: 0.9469, Recall: 0.9481, F1-score: 0.9471

# K-Nearest Neighbors with balanced data and no feature selection
name = "K-Nearest Neighbors with balanced data and no feature selection"
knn_balanced_noFS = KNeighborsClassifier(n_neighbors=10)
knn_balanced_noFS.fit(X_train_balanced, y_train_balanced)
evaluate_model_performance(knn_balanced_noFS, X_test, y_test, name)

# K-Nearest Neighbors with unbalanced data and no feature selection
name = "K-Nearest Neighbors with unbalanced data and no feature selection"
knn_unbalanced_noFS = KNeighborsClassifier(n_neighbors=10)
knn_unbalanced_noFS.fit(X_train, y_train)
evaluate_model_performance(knn_unbalanced_noFS, X_test, y_test, name)

#Evaluating best model-found with 5-fold cross-validation
# K-Nearest Neighbors with balanced data and feature selection
name = " K-Nearest Neighbors with balanced data and feature selection"
evaluate_model(knn_balanced_FS, X_valid_knn_balanced, y_valid, name)