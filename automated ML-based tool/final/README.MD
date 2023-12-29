# Dataset Analysis
This project involves the analysis of any dataset using machine learning techniques. The code is structured into several steps, including data loading, normalization, feature selection, cross-validation, model training, and performance evaluation. Additionally, visualizations are generated to provide insights into the dataset and its properties.

## Introduction
This project revolves around applying various operations on a biological dataset. The objective is to observe the changes and update the changes accordingly.

## Introduction
This project focuses on the comprehensive analysis of a biological dataset utilizing machine learning techniques. The objective is to observe patterns, derive insights, and enhance model performance through various operations.

## Getting Started
### Dependencies:
- `accuracy_score`, `classification_report`
- `train_test_split`, `cross_val_score`
- `learning_curve`
- `confusion_matrix`
- `numpy`
- `seaborn` as `sns`
- `pandas` as `pd`
- `matplotlib.pyplot` as `plt`
- `PdfPages`

### Custom Dependencies:
- `load_file` from `load.py`
- `normalized` from `normalization.py`
- `select` from `feature_selection.py`
- `cross` from `cross_validation.py`
- `mdl` from `model.py`

### Setup:
1. Install the required libraries.
2. Ensure custom libraries are in the same folder.
3. Run the model from `Main.py`.
4. Enter the relative path of the dataset when prompted (ensure the dataset is in matrix form with the target column at the end).

## Data Loading
Load the dataset using the `load_file` function:
```python
from load import load_file
X, y = load_file()
```
Datasets available for testing: Breast Cancer, Children Anemia, Diabetes, Faults, Iris (all in the same folder in a zip file).

## Normalization
Normalize the data using one of three techniques:
1. Standard Scaler
2. Min-Max Scaler
3. Robust Scaler
```python
from normalization import normalized
X_normalized, y = normalized(X, y)
```

## Feature Selection
Choose from various feature selection techniques:
1. SelectKBest with f_classif
2. SelectKBest with mutual_info_classif
3. Recursive Feature Elimination (RFE) with Random Forest
4. Recursive Feature Elimination (RFE) with SVC
5. Select From Model with Ridge
```python
from feature_selection import select
selected_feature = select(X_normalized, y)
```
Handle boundary conditions based on the dataset and features.

## Cross-Validation
Utilize one of three cross-validation techniques:
1. K-Fold Cross Validation
2. Stratified K-Fold Cross Validation
3. Leave-One-Out Cross Validation
```python
from cross_validation import cross
cv = cross(selected_feature, y)
```

## Model Selection
Choose from various models for binary or multiclass classification:
1. LogisticRegression()
2. RandomForestClassifier()
3. GradientBoostingClassifier()
4. SVC()
5. KNeighborsClassifier()
```python
from model import mdl
models = mdl(cv)
```

## Model Training
Evaluate model performance using train-test split and the `train_test_split` function from scikit-learn.

### Visualizations
Generate various visualizations:
- Boxplot of feature distribution before normalization
- Boxplot of feature distribution after normalization
- Pairplot of selected features, highlighting the target variable
- Bar chart depicting feature importance (if applicable)
- Countplot showing the distribution of the target variable
- Heatmap of the confusion matrix for model evaluation
- Heatmap illustrating the correlation matrix after normalization
- Bar chart showing classification metrics by class (Precision, Recall, F1-score).

# Outputs
All plots are saved in a PDF format, and command outputs are stored in `result.txt`.

# Conclusion
The analysis is completed with the message "Pdf is created successfully." Fine-tune the README.md file for better clarity and attractiveness.