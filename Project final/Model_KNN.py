("***KNN***")
# Author: Joelle Schiffmann

# Feature Selection for balanced & unbalanced data with SelectKBest
kbest_selector = SelectKBest(score_func=f_classif, k=10)


# Import from "Preprocessing.py"
from Preprocessing import *
#Import the evaulation function
from Model_evaluation import *


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





"""# Vorhersagen auf den Validierungsdaten
y_valid_pred_normal = knn_normal.predict(X_MI_valid)

# Metriken auf den Validierungsdaten berechnen
accuracy_normal = accuracy_score(y_valid, y_valid_pred_normal)
precision_normal = precision_score(y_valid, y_valid_pred_normal, average='weighted')
recall_normal = recall_score(y_valid, y_valid_pred_normal, average='weighted')
f1_normal = f1_score(y_valid, y_valid_pred_normal, average='weighted')

# ROC-AUC auf den Validierungsdaten berechnen
y_valid_prob_normal = knn_normal.predict_proba(X_MI_valid)
roc_auc_normal = roc_auc_score(pd.get_dummies(y_valid), y_valid_prob_normal, average='weighted')

# Ausgabe der Metriken
print("Validation Metrics with Normal Data:")
print(f"Accuracy: {accuracy_normal}")
print(f"Precision: {precision_normal}")
print(f"Recall: {recall_normal}")
print(f"F1-score: {f1_normal}")
print(f"ROC-AUC: {roc_auc_normal}")

# Plot der ROC-Kurve
plt.figure(figsize=(8, 6))
for i in range(y_valid_prob_normal.shape[1]):
    fpr, tpr, _ = roc_curve(y_valid, y_valid_prob_normal[:, i], pos_label=i)
    plt.plot(fpr, tpr, label=f'Class {i+1}')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Normal Data)')
plt.legend()
plt.show()

# Konfusionsmatrix
conf_matrix_normal = confusion_matrix(y_valid, y_valid_pred_normal)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_normal, annot=True, fmt='d', cmap='Blues', cbar=False)

# Annotieren der Achsen
class_names = ['Normal', 'Suspect', 'Pathological']
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks + 0.5, class_names)
plt.yticks(tick_marks + 0.5, class_names)

plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Normal Data)')
plt.show()

# Testen des Modells auf den Testdaten
y_test_pred_normal = knn_normal.predict(X_MI_test)

# Metriken auf den Testdaten berechnen
test_accuracy_normal = accuracy_score(y_test, y_test_pred_normal)
test_precision_normal = precision_score(y_test, y_test_pred_normal, average='weighted')
test_recall_normal = recall_score(y_test, y_test_pred_normal, average='weighted')
test_f1_normal = f1_score(y_test, y_test_pred_normal, average='weighted')

# ROC-AUC auf den Testdaten berechnen
y_test_prob_normal = knn_normal.predict_proba(X_MI_test)
test_roc_auc_normal = roc_auc_score(pd.get_dummies(y_test), y_test_prob_normal, average='weighted')

# Ausgabe der Metriken auf den Testdaten
print("\nTest Metrics with Normal Data:")
print(f"Accuracy: {test_accuracy_normal}")
print(f"Precision: {test_precision_normal}")
print(f"Recall: {test_recall_normal}")
print(f"F1-score: {test_f1_normal}")
print(f"ROC-AUC: {test_roc_auc_normal}")

("***OVERSAMPLED DATA***")
# Oversampling
ros = RandomOverSampler(random_state=42)
X_MI_train_oversampled, y_train_oversampled = ros.fit_resample(X_MI_train, y_train)
print('Oversampled training set size: {}'.format(len(X_MI_train_oversampled)))

# KNN-Modell mit oversampelten Daten erstellen und trainieren
knn_oversampled = KNeighborsClassifier()
knn_oversampled.fit(X_MI_train_oversampled, y_train_oversampled)

# Vorhersagen auf den Validierungsdaten
y_valid_pred_oversampled = knn_oversampled.predict(X_MI_valid)

# Metriken auf den Validierungsdaten berechnen
accuracy_oversampled = accuracy_score(y_valid, y_valid_pred_oversampled)
precision_oversampled = precision_score(y_valid, y_valid_pred_oversampled, average='weighted')
recall_oversampled = recall_score(y_valid, y_valid_pred_oversampled, average='weighted')
f1_oversampled = f1_score(y_valid, y_valid_pred_oversampled, average='weighted')

# ROC-AUC auf den Validierungsdaten berechnen
y_valid_prob_oversampled = knn_oversampled.predict_proba(X_MI_valid)
roc_auc_oversampled = roc_auc_score(pd.get_dummies(y_valid), y_valid_prob_oversampled, average='weighted')

# Ausgabe der Metriken
print("Validation Metrics with Oversampled Data:")
print(f"Accuracy: {accuracy_oversampled}")
print(f"Precision: {precision_oversampled}")
print(f"Recall: {recall_oversampled}")
print(f"F1-score: {f1_oversampled}")
print(f"ROC-AUC: {roc_auc_oversampled}")

# Plot der ROC-Kurve
plt.figure(figsize=(8, 6))
for i in range(y_valid_prob_oversampled.shape[1]):
    fpr, tpr, _ = roc_curve(y_valid, y_valid_prob_oversampled[:, i], pos_label=i)
    plt.plot(fpr, tpr, label=f'Class {i+1}')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Oversampled Data)')
plt.legend()
plt.show()

# Konfusionsmatrix
conf_matrix_oversampled = confusion_matrix(y_valid, y_valid_pred_oversampled)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_oversampled, annot=True, fmt='d', cmap='Blues', cbar=False)

# Annotieren der Achsen
class_names = ['Normal', 'Suspect', 'Pathological']
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks + 0.5, class_names)
plt.yticks(tick_marks + 0.5, class_names)

plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Oversampled Data)')
plt.show()

# Testen des Modells auf den Testdaten
y_test_pred_oversampled = knn_oversampled.predict(X_MI_test)

# Metriken auf den Testdaten berechnen
test_accuracy_oversampled = accuracy_score(y_test, y_test_pred_oversampled)
test_precision_oversampled = precision_score(y_test, y_test_pred_oversampled, average='weighted')
test_recall_oversampled = recall_score(y_test, y_test_pred_oversampled, average='weighted')
test_f1_oversampled = f1_score(y_test, y_test_pred_oversampled, average='weighted')

# ROC-AUC auf den Testdaten berechnen
y_test_prob_oversampled = knn_oversampled.predict_proba(X_MI_test)
test_roc_auc_oversampled = roc_auc_score(pd.get_dummies(y_test), y_test_prob_oversampled, average='weighted')

# Ausgabe der Metriken auf den Testdaten
print("\nTest Metrics with Oversampled Data:")
print(f"Accuracy: {test_accuracy_oversampled}")
print(f"Precision: {test_precision_oversampled}")
print(f"Recall: {test_recall_oversampled}")
print(f"F1-score: {test_f1_oversampled}")
print(f"ROC-AUC: {test_roc_auc_oversampled}")"""

# Evaluate models

#Evaluate without oversampling 
#with test data
evaluate_model(knn_normal, X_MI_test, y_test,data_type="test")
#with validation data
evaluate_model(knn_normal, X_MI_valid, y_test,data_type="validation")

#Evaluate with oversampling
evaluate_model(knn_oversampled, X_MI_test,y_test, oversampled=True)
