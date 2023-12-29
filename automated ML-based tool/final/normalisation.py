from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import pandas as pd

# Normalize/Standardize data

def normalised(X,y):
    print("\nChoose normalization/standardization technique:")
    print("1. Standard Scaler")
    print("2. Min-Max Scaler")
    print("3. Robust Scaler")
    # choice = int(input("Enter a choice"))
    
    try:
        choice = int(input("Enter your choice (1/2/3): "))
        if choice == 1:
            scaler = StandardScaler()
        elif choice == 2:
            scaler = MinMaxScaler()
        elif choice == 3:
            scaler = RobustScaler()
        else:
            raise ValueError("Invalid choice. Please enter 1, 2, or 3.")
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)

    X_scaled = scaler.fit_transform(X)
    normalised = pd.DataFrame( X_scaled,columns=X.columns)
    return normalised, y