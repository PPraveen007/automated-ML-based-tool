from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC


def mdl(problem_type):

    models_binary = {
        1: LogisticRegression(),
        2: RandomForestClassifier(),
        3: GradientBoostingClassifier(),
        4: SVC(),
        5: KNeighborsClassifier()
    }

    models_multiclass = {
        1: LogisticRegression(),
        2: RandomForestClassifier(),
        3: GradientBoostingClassifier(),
        4: SVC(),
        5: KNeighborsClassifier()
    }

    models_to_use = models_binary if problem_type == "Binary" else models_multiclass

    print("\nChoose a machine learning model:")
    for i, (model_number, model) in enumerate(models_to_use.items()):
        print(f"Model {model_number}: {type(model).__name__}")

    try:
        choice = int(input("Enter your choice (1/2/3/4/5): "))
        model = models_to_use[choice]
    except (ValueError, KeyError) as e:
        print(f"Error: Invalid choice. Please enter a number between 1 and 5.")
        exit(1)
    
    return model


