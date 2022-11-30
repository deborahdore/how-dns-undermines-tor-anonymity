from loguru import logger
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import classification_report, precision_recall_curve, PrecisionRecallDisplay
from sklearn.pipeline import Pipeline, FeatureUnion

from src.classification.ngrams_extractor import NgramsExtractor
from src.classification.const.const import RANDOM_FOREST_FILE, KNEIGHBORS_FILE, OUTPUT_REPORT, ROC_CURVE_FILE, \
    PR_CURVE_FILE
from src.classification.utils.utility import load_model


def eval(X_train, X_test, y_train, y_test, model_type, ngrams=1):
    """
    It loads the best model for the given model type and ngrams, and then evaluates it on the given test set

    :param X_train: The training data
    :param X_test: The test set
    :param y_train: The training labels
    :param y_test: the actual labels of the test set
    :param model_type: "RF" or "KNN"
    :param ngrams: the number of ngrams to use in the model, defaults to 1 (optional)
    """

    if model_type == "RF":
        path = RANDOM_FOREST_FILE.format(ngrams=ngrams)
    else:
        path = KNEIGHBORS_FILE.format(ngrams=ngrams)

    best_model = load_model(path)

    evaluation_pipeline = Pipeline([
        ('features', FeatureUnion([
            ('ngrams', NgramsExtractor(max_ngram_len=ngrams)),
        ])),
        ('rf', best_model)
    ])

    evaluation_pipeline.fit(X_train, y_train)
    y_pred = evaluation_pipeline.predict(X_test)
    y_score = evaluation_pipeline.predict_proba(X_test)

    # CLASSIFICATION REPORT
    report = classification_report(y_test, y_pred, target_names=['monitored', 'unmonitored'])
    with open(OUTPUT_REPORT.format(name=model_type, ngrams=ngrams), "w") as f:
        f.write(report)

    # ROC CURVE
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score[:, 1])
    roc_auc = metrics.auc(fpr, tpr)
    logger.info(f"Area under the ROC curve: {roc_auc}")
    metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot()
    plt.savefig(ROC_CURVE_FILE.format(name=model_type, ngrams=ngrams), dpi=1200)
    plt.close()

    # PRECISION-RECALL CURVE
    precision, recall, threshold = precision_recall_curve(y_test, y_score[:, 1])
    PrecisionRecallDisplay(precision=precision, recall=recall).plot()
    plt.savefig(PR_CURVE_FILE.format(name=model_type, ngrams=ngrams), dpi=1200)
    plt.close()
