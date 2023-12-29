from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages 
import builtins 

# composed dependency
from load import load_file
from normalisation import normalised
from feature_selection import select
from cross_validation import cross
from model import mdl

with open('results_cs21b1035.txt', 'w') as file:
    # Redirect print statements to the file
    def custom_print(*args, **kwargs):
        file.write(' '.join(map(str, args)) + '\n')
        builtins.print(*args, **kwargs)

    print = custom_print
        
    #loading data    
    X, y = load_file()
    print(X)

    #For finding the data is binary or multiclass
    no_of_classes= len(np.unique(y))
    problem_type = "Binary " if no_of_classes == 2 else " multiclass"
    print(f"\nThis is a {problem_type} classification problem.") 

    # Normalisation
    X_normalized, y = normalised(X,y)
    print("Normalized Data:")
    print(X_normalized) 

    # features = fetureselection()
    selected_feature = select(X_normalized,y)
    print("selected feature:")
    print(selected_feature)

    #corss_validation
    cv = cross(selected_feature, y)
    print(cv)

    # Model selection{on basis of binary and multiclass}
    models = mdl(problem_type)

    # Train-test split for validation
    X_train, X_test, y_train, y_test = train_test_split(selected_feature, y, test_size=0.1, random_state=42)

    # Train the selected model
    models.fit(X_train, y_train)

    # Validate using a blind dataset
    y_pred = models.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy on Blind Dataset: {accuracy:.2f}\n")

    # Cross-validation performance metrics
    cv_scores = cross_val_score(models, selected_feature, y, cv=cv, scoring='accuracy')
    print(f"Cross-Validation Accuracy: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})\n")

    #classification report
    with np.errstate(divide ="warn", invalid="warn"):
        classification_report_str = classification_report(y_test,y_pred,zero_division=1)
        print(classification_report_str)

# Reset the print function to the built-in print
print = builtins.print
#========================================PLOTTING========================================================

# #Visualizations
#pdf genrator
pdf_filename = "Praveen_Raj_cs21b1035.pdf"
with PdfPages(pdf_filename) as pdf:

    # Second page - Before Normalization
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=X, orient='h', ax=ax)
    ax.set_title("Before Normalization")
    pdf.savefig()
    plt.close(fig)

    # Third page - After Normalization
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=X_normalized, orient='h', ax=ax)
    ax.set_title("After Normalization")
    pdf.savefig()
    plt.close(fig)

    # Fourth page - Feature Selection
    pairplot = sns.pairplot(selected_feature.join(pd.Series(y, name='target')), hue='target')
    pairplot.fig.suptitle("Feature Selection", y=1.02)  # Add a title outside the plot area
    ax.set_title("Pair plot")
    pdf.savefig()
    plt.close(pairplot.fig)

    # fift page- Feature Importance Plot (if applicable)
    if hasattr(models, 'feature_importances_'):
        feature_importance = models.feature_importances_
        feature_names = selected_feature.columns[:len(feature_importance)]
        plt.barh(feature_names, feature_importance)
        plt.xlabel('Feature Importance')
        plt.title('Feature Importance Plot')
        pdf.savefig()
        plt.close(fig)
        
    #sixth page- distribbution plot
    sns.countplot(x=y)
    plt.title("Distribution of Target Variable")
    pdf.savefig()
    plt.close(fig)

    #sevent page- Plot confusion matrix as heatmap
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    pdf.savefig()
    plt.close(fig)
    
    #eight page -correlation matrix
    correlation_matrix = X_normalized.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    pdf.savefig()
    plt.close(fig)
    
    # Extract metrics and labels
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=1)
    class_labels = list(report.keys())[:-3] 
    metrics = list(report[class_labels[0]].keys())
    plt.figure(figsize=(10, 6))
    bar_width = 0.2
    for i, metric in enumerate(metrics):
        values = [report[label][metric] for label in class_labels]
        positions = np.arange(len(class_labels)) + i * bar_width
        plt.bar(positions, values, width=bar_width, label=metric)
    plt.title(f'{models} Metrics by Class')
    plt.xlabel('Class')
    plt.ylabel('Values')
    plt.xticks(np.arange(len(class_labels)) + bar_width, class_labels)
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
    pdf.savefig()
    plt.close(fig)
    
print("Pdf is created succesfully")