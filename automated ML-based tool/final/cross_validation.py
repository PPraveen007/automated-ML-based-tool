from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, LeaveOneOut

def cross(X,y):
    # Cross-validation
    print("\nChoose cross-validation technique:")
    print("1. K-Fold Cross Validation")
    print("2. Stratified K-Fold Cross Validation")
    print("3. Leave-One-Out Cross Validation")

    try:
        choice = int(input("Enter your choice (1/2/3): "))
        if choice == 1:
            cv_method = KFold(n_splits=5, shuffle=True, random_state=42)
        elif choice == 2:
            cv_method = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        elif choice == 3:
            cv_method = LeaveOneOut()
        else:
            raise ValueError("Invalid choice. Please enter 1, 2, or 3.")
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)
        
    return cv_method