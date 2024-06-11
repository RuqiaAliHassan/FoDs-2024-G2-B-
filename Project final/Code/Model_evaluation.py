("***EVALUATING MODELS***")
#import liberaries
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

("***EVALUATION FUNCTION***")
def evaluate_model_performance(model_name, X, y, name):
    y_pred = model_name.predict(X)
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='weighted')
    recall = recall_score(y, y_pred, average='weighted')
    f1 = f1_score(y, y_pred, average='weighted')

    print("Performance of", name)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("")

("***COMPARING BEST MODELS***")
def evaluate_model(model, X_valid, y_valid, name):
    y_pred = model.predict(X_valid)
    print(name)
    print(classification_report(y_valid, y_pred))

    conf_matrix = confusion_matrix(y_valid, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Suspect', 'Pathological'],
                yticklabels=['Normal', 'Suspect', 'Pathological'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title("Confusion Matrix for " + name)
    plt.savefig("../output/confusion_matrix_{}.png".format(name))
    plt.show()

