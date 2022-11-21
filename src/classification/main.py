import warnings

import matplotlib.pyplot as plt
from classification.utils.path import RESULTS_DIR, RANDOM_FOREST_FILE, \
    OUTPUT_REPORT, ROC_CURVE_FILE, PR_CURVE_FILE
from classification.utils.utility import *
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import FeatureUnion, Pipeline

from src.classification.feature_extraction import NgramsExtractor

warnings.filterwarnings("ignore")


def create_search_cv():
    """
    It creates a randomized search cross validation object that will be used to find the best hyperparameters for the random
    forest model
    :return: A randomized search cross validation object
    """
    # num of trees
    n_estimators = [int(x) for x in np.linspace(start=200, stop=500, num=10)]
    # Number of features to consider at every split
    max_features = ["sqrt", "log2"]
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    rf = RandomForestClassifier(n_jobs=-1)
    # 5 folds
    rf_random = RandomizedSearchCV(estimator=rf,
                                   param_distributions=random_grid,
                                   n_iter=100, cv=5,
                                   verbose=2,
                                   random_state=42,
                                   n_jobs=-1)
    return rf_random


def cross_validation(model, X, y, cv=5):
    """
    It takes a model, X, y, and cv as parameters and returns a dictionary of the training and validation scores for
    accuracy, precision, recall, and f1

    :param model: The model you want to use for cross validation
    :param X: The data to fit. Can be for example a list, or an array at least 2d
    :param y: The target variable
    :param cv: Number of folds to use for cross-validation, defaults to 5 (optional)
    :return: A dictionary with the training and validation scores for each of the metrics.
    """
    scoring = ['accuracy', 'precision', 'recall', 'f1']
    results = cross_validate(estimator=model,
                             X=X,
                             y=y,
                             cv=cv,
                             scoring=scoring,
                             return_train_score=True,
                             verbose=2)

    return {"Training Accuracy scores": results['train_accuracy'],
            "Mean Training Accuracy": results['train_accuracy'].mean() * 100,
            "Training Precision scores": results['train_precision'],
            "Mean Training Precision": results['train_precision'].mean(),
            "Training Recall scores": results['train_recall'],
            "Mean Training Recall": results['train_recall'].mean(),
            "Training F1 scores": results['train_f1'],
            "Mean Training F1 Score": results['train_f1'].mean(),
            "Validation Accuracy scores": results['test_accuracy'],
            "Mean Validation Accuracy": results['test_accuracy'].mean() * 100,
            "Validation Precision scores": results['test_precision'],
            "Mean Validation Precision": results['test_precision'].mean(),
            "Validation Recall scores": results['test_recall'],
            "Mean Validation Recall": results['test_recall'].mean(),
            "Validation F1 scores": results['test_f1'],
            "Mean Validation F1 Score": results['test_f1'].mean()
            }


def evaluation(X, y):
    """
    It loads the best model from the randomized grid search, creates a new pipeline without the randomized grid search, and
    then evaluates the model on the test set

    :param X: the dataframe containing the features
    :param y: The target variable
    """

    # create results dir
    if not os.path.isdir(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    # load model
    best_model = joblib.load(RANDOM_FOREST_FILE)

    # create new pipeline without the randomized grid search
    evaluation_pipeline = Pipeline([
        ('features', FeatureUnion([
            ('ngrams', NgramsExtractor(max_ngram_len=1)),
        ])),
        ('rf', best_model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True)

    evaluation_pipeline.fit(X_train, y_train)

    # Calculating the scores for the training and validation sets.
    scores = cross_validation(evaluation_pipeline, X_train, y_train)
    print(scores)

    # probabilities of choosing one or the other class
    y_proba = evaluation_pipeline.predict_proba(X_test)
    auroc = roc_auc_score(y_test.flatten(), y_proba[:, 1])
    print("Area under the ROC curve:", auroc)

    # Predicting the labels of the test set.
    y_pred = evaluation_pipeline.predict(X_test)

    report = classification_report(y_test, y_pred, target_names=['monitored', 'unmonitored'], output_dict=True)

    # write results
    with open(OUTPUT_REPORT, "w") as f:
        for key, value in scores.items():
            f.write('%s:%s\n' % (key, value))
        for key, value in report.items():
            f.write('%s:%s\n' % (key, value))
        f.write(f'Auroc: {auroc}')

    metrics.plot_roc_curve(evaluation_pipeline, X_test, y_test, drop_intermediate=False)
    # plt.show()
    plt.savefig(ROC_CURVE_FILE, dpi=1200)

    metrics.plot_precision_recall_curve(evaluation_pipeline, X_test, y_test)
    # plt.show()
    plt.savefig(PR_CURVE_FILE, dpi=1200)


def ow_experiment(X, y):
    """
    It loads the dataset, splits it into training and test sets, creates a pipeline with a feature union and a random search
    cross validation, trains the model, saves the best model, evaluates the model and plots the ROC and precision-recall
    curves

    Parameters
    ----------
    :param X: the data
    :param y: The target variable
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True)

    rf_random = create_search_cv()

    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('ngrams', NgramsExtractor(max_ngram_len=1)),
        ])),
        ('rs_cv', rf_random)
    ])

    print("Searching for the best combination of parameters")
    pipeline.fit(X_train, y_train.ravel())

    print("Saving best model")
    save_model(pipeline['rs_cv'].best_estimator_, RANDOM_FOREST_FILE)

    # Best params: {'n_estimators': 733, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt',
    # 'max_depth': 40, 'bootstrap': False}
    print(f"Best params: {pipeline['rs_cv'].best_params_}")


if __name__ == '__main__':
    print("Load Dataset")
    X, y = load_dataset()

    print("Starting experiment")
    ow_experiment(X, y)
    print("Evaluation")
    evaluation(X, y)
