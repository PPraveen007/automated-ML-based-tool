import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


def load_file():
    file_path = input("Enter the path of your dataset: ")
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        print("File not found. Please provide the correct file path.")
        exit(1)

    target_variable = data.columns[-1]
    print(target_variable)

    # Exclude 'ID' column from X
    if 'ID' in data.columns:
        X = data.drop(['ID', target_variable], axis=1)
    else:
        X = data.drop(target_variable, axis=1)

    y = data[target_variable]

    # Handle 'most_frequent' imputation
    imputer = SimpleImputer(strategy="most_frequent")
    
    # Impute missing values only for columns with missing values
    cols_with_missing_values = X.columns[X.isnull().any()].tolist()
    if cols_with_missing_values:
        X[cols_with_missing_values] = imputer.fit_transform(X[cols_with_missing_values])

    if y.dtype == 'O':
        label_encoder_y = LabelEncoder()
        y = label_encoder_y.fit_transform(y)

    for column in X.columns:
        if X[column].dtype == 'O' and not X[column].apply(lambda x: isinstance(x, (int, float))).all():
            label_encoder_X = LabelEncoder()
            X[column] = label_encoder_X.fit_transform(X[column])

    return X, pd.Series(y, name=target_variable)
