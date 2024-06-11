#Author: Ruqia Ali Hassan
#import libraries

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#Import from "Preprocessing.py"
from Preprocessing import *
#Import the evaulation function
from Model_evaluation import *

# Feature selection for balanced & unbalanced data with Random Forest
forest_selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))
# For unbalanced data
forest_selector.fit(X_train, y_train)
X_train_forest_unbalanced = forest_selector.transform(X_train)
X_test_forest_unbalanced = forest_selector.transform(X_test)
X_valid_forest_unbalanced = forest_selector.transform(X_valid)
selected_features_forest_unbalanced = X_train.columns[forest_selector.get_support()]
print("Remaining features after Random Forest feature selection with unbalanced data:",{len(selected_features_forest_unbalanced)})
print(selected_features_forest_unbalanced)
#7 remained: 'prolonged_decelerations', 'pc_short_term_ab_variability', 'mean_short_term_variability', 'pc_long_term_ab_variability', 'histogram_mode', 'histogram_mean', 'histogram_median'

# For balanced data
forest_selector.fit(X_train_balanced, y_train_balanced)
X_train_forest_balanced = forest_selector.transform(X_train_balanced)
X_test_forest_balanced = forest_selector.transform(X_test)
X_valid_forest_balanced = forest_selector.transform(X_valid)
selected_features_forest_balanced = X_train.columns[forest_selector.get_support()]
print("Remaining features after Random Forest feature selection with balanced data:",{len(selected_features_forest_balanced)})
print(selected_features_forest_balanced)
#8 remained: 'accelerations', 'pc_short_term_ab_variability', 'mean_short_term_variability', 'pc_long_term_ab_variability', 'mean_long_term_variability', 'histogram_mode', 'histogram_mean', 'histogram_median'

# Random Forest with balanced data and feature selection
name = "Random Forest with balanced data and feature selection"
RF_balanced_FS = RandomForestClassifier(n_estimators=100, random_state=42)
RF_balanced_FS.fit(X_train_forest_balanced, y_train_balanced)
evaluate_model_performance(RF_balanced_FS, X_test_forest_balanced, y_test, name)

# Random Forest with unbalanced data and feature selection
name = "Random Forest with unbalanced data and feature selection"
RF_unbalanced_FS = RandomForestClassifier(n_estimators=100, random_state=42)
RF_unbalanced_FS.fit(X_train_forest_unbalanced, y_train)
evaluate_model_performance(RF_unbalanced_FS, X_test_forest_unbalanced, y_test, name)

# Random Forest with balanced data and no feature selection
name = "Random Forest with balanced data and no feature selection"
RF_balanced_noFS = RandomForestClassifier(n_estimators=100, random_state=42)
RF_balanced_noFS.fit(X_train_balanced, y_train_balanced)
evaluate_model_performance(RF_balanced_noFS, X_test, y_test, name)

# Random Forest with unbalanced data and no feature selection
name = "Random Forest with unbalanced data and no feature selection"
RF_unbalanced_noFS = RandomForestClassifier(n_estimators=100, random_state=42)
RF_unbalanced_noFS.fit(X_train, y_train)
evaluate_model_performance(RF_unbalanced_noFS, X_test, y_test, name)
#best Performance in all metrics: Accuracy:  0.9764, Precision: 0.9763, Recall: 0.9764, F1-score: 0.9760


#Evaluating best model-found with 5-fold cross-validation
#Random Forest with balanced data and feature selection
name = " Random Forest with balanced data and feature selection"
evaluate_model(RF_balanced_FS, X_valid_forest_balanced, y_valid, name)
#still the best performance

("***FEATURE IMPORTANCE***")

#Balanced
# Extract feature importance values
forest_selector = RandomForestClassifier(n_estimators=100, random_state=42)
# For unbalanced data
forest_selector.fit(X_train_balanced, y_train_balanced)

feature_importance_all = forest_selector.feature_importances_

# Create a DataFrame to store feature importance values
data_rf_feature_importance_all = pd.DataFrame({'Features': X_train.columns, 'Importance': feature_importance_all})

# Sort the DataFrame by importance values
data_rf_feature_importance_all.sort_values(by='Importance', ascending=False, inplace=True)

# Visualize feature importance
fig, ax = plt.subplots(figsize=(9,9))
sns.barplot(x='Importance', y='Features', data=data_rf_feature_importance_all, ax=ax, palette='coolwarm')
ax.set_title('Feature importance | Random Forest | All Features-Balanced')
ax.set_xlabel('Importance')
ax.set_ylabel('Features')
plt.tight_layout()
plt.savefig("../output/feature_importance_random_forest_all_features-balanced.png")
plt.show()


#Unbalanced
# Extract feature importance values
forest_selector = RandomForestClassifier(n_estimators=100, random_state=42)
# For unbalanced data
forest_selector.fit(X_train, y_train)

feature_importance_all = forest_selector.feature_importances_

# Create a DataFrame to store feature importance values
data_rf_feature_importance_all = pd.DataFrame({'Features': X_train.columns, 'Importance': feature_importance_all})

# Sort the DataFrame by importance values
data_rf_feature_importance_all.sort_values(by='Importance', ascending=False, inplace=True)

# Visualize feature importance
fig, ax = plt.subplots(figsize=(9,9))
sns.barplot(x='Importance', y='Features', data=data_rf_feature_importance_all, ax=ax, palette='coolwarm')
ax.set_title('Feature importance | Random Forest | All Features-Unbalanced')
ax.set_xlabel('Importance')
ax.set_ylabel('Features')
plt.tight_layout()
plt.savefig("../output/feature_importance_random_forest_all_features-unbalanced.png")
plt.show()