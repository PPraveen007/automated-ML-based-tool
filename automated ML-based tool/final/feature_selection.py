# Feature Selection
from sklearn.feature_selection import SelectKBest, f_classif, chi2, RFE ,SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.svm import SVC
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import Ridge

def select(X_scaled, y):
    print("\nChoose feature selection technique:")
    print("1. SelectKBest with f_classif")
    print("2. SelectKBest with mutual_info_classif")
    print("3. Recursive Feature Elimination (RFE) with Random Forest ")
    print("4. Recursive Feature Elimination (RFE) with SVC")
    print("5. Select From Model with Ridge")

    choice = int(input("Enter a choice ")) 
    try:
        if choice == 1:
            k = int(input("Enter the value of K: "))
            selector = SelectKBest(f_classif, k=k)
        elif choice == 2:
            k = int(input("Enter the value of K: "))
            selector = SelectKBest(mutual_info_classif, k=k)
        elif choice == 3:
            k = int(input("Enter the value of K: "))
            model = RandomForestClassifier()  # You can replace this with any other model
            selector = RFE(model, n_features_to_select=k)
        elif choice == 4:
            k = int(input("Enter the value of K: "))
            model = SVC(kernel='linear')  # You can replace this with any other SVM configuration
            selector = RFE(model, n_features_to_select=k)
        elif choice == 5:
            # Ridge-based feature selection
            alpha = float(input("Enter the alpha value for Ridge: "))
            model = Ridge(alpha=alpha)
            selector = SelectFromModel(model)
        else:
            raise ValueError("Invalid choice. Please enter 1, 2, 3, 4, or 5.")
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)

    X_selected = selector.fit_transform(X_scaled, y)
    selected_feature = pd.DataFrame( X_scaled,columns=X_scaled.columns[:X_selected.shape[1]])
    return selected_feature

