("***LOGISTIC REGRESSION***")
# Author: Lara Jenni

#importing libraries
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Import from "Preprocessing.py"
from Preprocessing import *

#Import the evaulation function
from Model_evaluation import *


#feature selection for balanced & unbalanced data with L1-regulation
logistic_selector = SelectFromModel(LogisticRegression(penalty='l1', C=0.01, solver='liblinear', random_state=42))
# For unbalanced data
logistic_selector.fit(X_train, y_train)
X_train_logistic_unbalanced = logistic_selector.transform(X_train)
X_test_logistic_unbalanced = logistic_selector.transform(X_test)
X_valid_logistic_unbalanced = logistic_selector.transform(X_valid)
selected_features_unbalanced = X_train.columns[logistic_selector.get_support()]
print("Remaining features after Logistic Regression feature selection with unbalanced data:")
print(selected_features_unbalanced)
#9 remained: 'baseline_value', 'accelerations', 'uterine_contractions', 'prolonged_decelerations', 'pc_short_term_ab_variability', 'mean_short_term_variability', 'pc_long_term_ab_variability', 'histogram_mode', 'histogram_mean'
# For balanced data
logistic_selector.fit(X_train_balanced, y_train_balanced)
X_train_logistic_balanced = logistic_selector.transform(X_train_balanced)
X_test_logistic_balanced = logistic_selector.transform(X_test)
X_valid_logistic_balanced = logistic_selector.transform(X_valid)
selected_features_balanced = X_train.columns[logistic_selector.get_support()]
print("Remaining features after Logistic Regression feature selection with balanced data:")
print(selected_features_balanced)
#14 remained: 'baseline_value', 'accelerations', 'uterine_contractions', 'light_decelerations', 'severe_decelerations', 'prolonged_decelerations', 'pc_short_term_ab_variability', 'mean_short_term_variability', 'pc_long_term_ab_variability', 'histogram_max', 'histogram_mode', 'histogram_mean', 'histogram_median', 'histogram_variance'

#Logist Regression with balanced data and feature selection
name = "Logist Regression with balanced data and feature selection"
LR_balanced_FS = LogisticRegression(penalty=None, multi_class='multinomial', random_state=42)
LR_balanced_FS.fit(X_train_logistic_balanced, y_train_balanced)
evaluate_model_performance(LR_balanced_FS, X_test_logistic_balanced, y_test, name)

#Logistic Regression with unbalanced data and feature selection
name = "Logistic Regression with unbalanced data and feature selection"
LR_unbalanced_FS = LogisticRegression(penalty=None, multi_class='multinomial', random_state=42)
LR_unbalanced_FS.fit(X_train_logistic_unbalanced, y_train)
evaluate_model_performance(LR_unbalanced_FS, X_test_logistic_unbalanced, y_test, name)

#Logistic Regression with balanced data and no feature selection
name = "Logistic Regression with balanced data and no feature selection"
LR_balanced_noFS = LogisticRegression(penalty=None, multi_class='multinomial', random_state=42)
LR_balanced_noFS.fit(X_train_balanced, y_train_balanced)
evaluate_model_performance(LR_balanced_noFS, X_test, y_test, name)

#Logistic Regression with unbalanced data and no feature selection
name = "Logistic Regression with unbalanced data and no feature selection"
LR_unbalanced_noFS = LogisticRegression(penalty=None, multi_class='multinomial', random_state=42)
LR_unbalanced_noFS.fit(X_train, y_train)
evaluate_model_performance(LR_unbalanced_noFS, X_test, y_test, name)
#best Performance in all metrics: Accuracy: 0.9104, Precision: 0.9149, Recall: 0.9104, F1-score: 0.9104


#Evaluating best model-found with 5-fold cross-validation
#Logist Regression with balanced data and feature selection
name = "Logist Regression with balanced data and feature selection"
evaluate_model(LR_balanced_FS, X_valid_logistic_balanced, y_valid, name)
