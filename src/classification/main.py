import warnings

import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_recall_curve, PrecisionRecallDisplay
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import FeatureUnion, Pipeline

from src.classification.feature_extraction import NgramsExtractor
from src.classification.utils.path import OUTPUT_REPORT, ROC_CURVE_FILE, PR_CURVE_FILE, KNEIGHBORS_FILE, \
    RANDOM_FOREST_FILE
from src.classification.utils.utility import *
from src.classification.utils.utility import load_model, create_random_grid

warnings.filterwarnings("ignore")


def evaluation(X_train, X_test, y_train, y_test, model_type):
    """
    It loads the best model, creates a new pipeline without the randomized grid search, fits the pipeline to the training
    data, predicts the labels of the test set, calculates the scores for the training and validation sets, and saves the
    scores to a file

    :param X_train: The training set of the features
    :param X_test: The test set
    :param y_train: The training labels
    :param y_test: The true labels of the test set
    """
    if model_type == "RF":
        path = RANDOM_FOREST_FILE
    else:
        path = KNEIGHBORS_FILE

    best_model = load_model(path)

    evaluation_pipeline = Pipeline([
        ('features', FeatureUnion([
            ('ngrams', NgramsExtractor(max_ngram_len=1)),
        ])),
        ('rf', best_model)
    ])

    evaluation_pipeline.fit(X_train, y_train)
    y_pred = evaluation_pipeline.predict(X_test)
    y_score = evaluation_pipeline.predict_proba(X_test)

    # CLASSIFICATION REPORT
    report = classification_report(y_test, y_pred, target_names=['monitored', 'unmonitored'])
    with open(OUTPUT_REPORT.format(name=model_type), "w") as f:
        f.write(report)

    # ROC CURVE
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score[:, 1])
    roc_auc = metrics.auc(fpr, tpr)
    logger.info(f"Area under the ROC curve: {roc_auc}")
    metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot()
    plt.savefig(ROC_CURVE_FILE.format(name=model_type), dpi=1200)
    plt.close()

    # PRECISION-RECALL CURVE
    precision, recall, threshold = precision_recall_curve(y_test, y_score[:, 1])
    PrecisionRecallDisplay(precision=precision, recall=recall).plot()
    plt.savefig(PR_CURVE_FILE.format(name=model_type), dpi=1200)
    plt.close()


def ow_experiment(X_train, y_train, model_type):
    """
    It creates a pipeline that first extracts features from the data, then uses a randomized search to find the best
    combination of parameters for a random forest classifier

    :param X_train: the training data
    :param y_train: the training labels
    """

    if model_type == "RF":
        model = RandomForestClassifier(n_jobs=-1)
        path = RANDOM_FOREST_FILE
    else:
        model = KNeighborsClassifier(n_jobs=-1)
        path = KNEIGHBORS_FILE

    random_grid = create_random_grid(model_type)

    rf_random = RandomizedSearchCV(estimator=model,
                                   param_distributions=random_grid,
                                   n_iter=200,
                                   cv=5,
                                   scoring="f1",  # both precision and recall should be maximised
                                   n_jobs=-1)

    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('ngrams', NgramsExtractor(max_ngram_len=1)),
        ])),
        ('rs_cv', rf_random)
    ])

    logger.info("Searching for the best combination of parameters")
    pipeline.fit(X_train, y_train.ravel())

    logger.info("Saving best model")
    save_model(pipeline['rs_cv'].best_estimator_, path)

    logger.info(f"Best params: {pipeline['rs_cv'].best_params_}")


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_split_dataset()

    logger.info("Starting experiment using a Random Forest Classifier")
    ow_experiment(X_train, y_train, "RF")

    logger.info("Starting evaluation using a Random Forest Classifier")
    evaluation(X_train, X_test, y_train, y_test, "RF")

    logger.info("Starting experiment using a K Nearest Neighbor")
    ow_experiment(X_train, y_train, "KNN")

    logger.info("Starting evaluation using a K Nearest Neighbor")
    evaluation(X_train, X_test, y_train, y_test, "KNN")

