("***EVALUATING MODELS***")
#import liberaries
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

("***EVALUATION FUNCTION***")

def evaluate_model_performance(model, X, y, name, n_splits=5):
    # Initialize Stratified K-Fold cross-validator
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Perform k-fold cross-validation and collect scores
    accuracy_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    precision_scores = cross_val_score(model, X, y, cv=skf, scoring='precision_weighted')
    recall_scores = cross_val_score(model, X, y, cv=skf, scoring='recall_weighted')
    f1_scores = cross_val_score(model, X, y, cv=skf, scoring='f1_weighted')

    # Print the mean and standard deviation of cross-validation scores
    print(f"Performance of {name} with {n_splits}-fold cross-validation:")
    print(f"Accuracy: {accuracy_scores.mean():.4f} ± {accuracy_scores.std():.4f}")
    print(f"Precision: {precision_scores.mean():.4f} ± {precision_scores.std():.4f}")
    print(f"Recall: {recall_scores.mean():.4f} ± {recall_scores.std():.4f}")
    print(f"F1-score: {f1_scores.mean():.4f} ± {f1_scores.std():.4f}")
    print("")

("***COMPARING BEST MODELS***")
def evaluate_model(model, X_valid, y_valid, name):
    y_pred = model.predict(X_valid)
    print(name)
    print(classification_report(y_valid, y_pred))


    #confusion matrix
    conf_matrix = confusion_matrix(y_valid, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Suspect', 'Pathological'],
                yticklabels=['Normal', 'Suspect', 'Pathological'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title("Confusion Matrix for " + name)
    plt.savefig("../output/confusion_matrix_{}.png".format(name))
    plt.show()

