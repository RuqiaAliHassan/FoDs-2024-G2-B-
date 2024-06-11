("***SVM***")
# Author: Thurga Pararajasingam

# importing libraries
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Import from "Preprocessing.py"
from Preprocessing import *

#Import the evaulation function
from Model_evaluation import *

# Feature selection for balanced & unbalanced data with L1-regularization
svm_selector = SelectFromModel(SVC(kernel='linear', C=0.01, random_state=42))

# For unbalanced data
svm_selector.fit(X_train, y_train)
X_train_svm_unbalanced = svm_selector.transform(X_train)
X_test_svm_unbalanced = svm_selector.transform(X_test)
X_valid_svm_unbalanced = svm_selector.transform(X_valid)
selected_features_svm_unbalanced = X_train.columns[svm_selector.get_support()]
print("Remaining features after SVM feature selection with unbalanced data:")
print(selected_features_svm_unbalanced)
#9 remained: 'baseline_value', 'accelerations', 'uterine_contractions', 'prolonged_decelerations', 'pc_short_term_ab_variability', 'pc_long_term_ab_variability', 'histogram_mean', 'histogram_median', 'histogram_variance'
# For balanced data
svm_selector.fit(X_train_balanced, y_train_balanced)
X_train_svm_balanced = svm_selector.transform(X_train_balanced)
X_test_svm_balanced = svm_selector.transform(X_test)
X_valid_svm_balanced = svm_selector.transform(X_valid)
selected_features_svm_balanced = X_train.columns[svm_selector.get_support()]
print("Remaining features after SVM feature selection with balanced data:")
print(selected_features_svm_balanced)
#10 remained: 'baseline_value', 'accelerations', 'uterine_contractions', 'light_decelerations', 'prolonged_decelerations', 'pc_short_term_ab_variability', 'pc_long_term_ab_variability', 'histogram_mean', 'histogram_median', 'histogram_variance'
# Support Vector Machine with balanced data and feature selection
name = "Support Vector Machine with balanced data and feature selection"
SVM_balanced_FS = SVC(kernel='linear', random_state=42)
SVM_balanced_FS.fit(X_train_svm_balanced, y_train_balanced)
evaluate_model_performance(SVM_balanced_FS, X_test_svm_balanced, y_test, name)

# Support Vector Machine with unbalanced data and feature selection
name = "Support Vector Machine with unbalanced data and feature selection"
SVM_unbalanced_FS = SVC(kernel='linear', random_state=42)
SVM_unbalanced_FS.fit(X_train_svm_unbalanced, y_train)
evaluate_model_performance(SVM_unbalanced_FS, X_test_svm_unbalanced, y_test, name)

# Support Vector Machine with balanced data and no feature selection
name = "Support Vector Machine with balanced data and no feature selection"
SVM_balanced_noFS = SVC(kernel='linear', random_state=42)
SVM_balanced_noFS.fit(X_train_balanced, y_train_balanced)
evaluate_model_performance(SVM_balanced_noFS, X_test, y_test, name)

# Support Vector Machine with unbalanced data and no feature selection
name = "Support Vector Machine with unbalanced data and no feature selection"
SVM_unbalanced_noFS = SVC(kernel='linear', random_state=42)
SVM_unbalanced_noFS.fit(X_train, y_train)
evaluate_model_performance(SVM_unbalanced_noFS, X_test, y_test, name)
#best Performance in all metrics: Accuracy:  0.9245, Precision: 0.9252, Recall: 0.9245, F1-score: 0.9244

#-> overall best performance with Random Forest with unba

#Evaluating Model
#Support Vector Machine with unbalanced data and no feature selection
name = "Support Vector Machine with unbalanced data and no feature selection"
evaluate_model(SVM_unbalanced_noFS, X_valid, y_valid, name)