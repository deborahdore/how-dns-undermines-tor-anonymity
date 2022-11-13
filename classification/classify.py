import gc

import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import label_binarize

from classification.Path import DATASET_CLOSED_WORLD, DATASET_OPEN_WORLD, OUTPUT_ACC, RESULTS_DIR
from classification.feature_extraction import NgramsExtractor
from utility import *


def classify(dataset, output_acc):
    """Function that runs the classification pipeline."""

    print(f'Dataset length: {dataset.shape[0]}')

    X = dataset[['lengths']]

    y = dataset[['target']]
    y = label_binarize(y, classes=['monitored', 'unmonitored'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True)

    combined_features = FeatureUnion([
        ('ngrams', NgramsExtractor(max_ngram_len=1)),
    ])

    pipeline = Pipeline([
        ('features', combined_features),
        ('clf', RandomForestClassifier(n_estimators=100))
    ])

    # Training with pipeline
    pipeline.fit(X_train, y_train)

    # Prediction
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("Accuracy Score:", acc)

    y_proba = pipeline.predict_proba(X_test)
    auroc = roc_auc_score(y_test.flatten(), y_proba[:, 1])
    print("Area under the ROC curve:", auroc)

    with open(output_acc, "w") as f:
        f.write("Accuracy Score: " + str(acc))
        f.write("\n")
        f.write("AUROC: " + str(auroc))

    metrics.plot_roc_curve(pipeline, X_test, y_test, drop_intermediate=False)
    #plt.show()
    plt.savefig(RESULTS_DIR + "/roc_curve.svg", dpi=1200)

    metrics.plot_precision_recall_curve(pipeline, X_test, y_test)
    #plt.show()
    plt.savefig(RESULTS_DIR + "/precision_recall_curve.svg", dpi=1200)


def ow_experiment():
    """ Function to run open world experiment."""

    # create results dir
    if not os.path.isdir(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    # open world data
    df_monitored = load_data(DATASET_CLOSED_WORLD)
    df_monitored['target'] = "monitored"

    df_unmonitored = load_data(DATASET_OPEN_WORLD)
    df_unmonitored['target'] = "unmonitored"

    print(f'Size monitored dataset: {df_monitored.shape[0]}')
    print(f'Size unmonitored dataset: {df_unmonitored.shape[0]}')

    df_monitored_sample = df_monitored.sample(df_unmonitored.shape[0])

    dataset = pd.concat([df_monitored_sample, df_unmonitored], axis=0)

    del df_monitored
    del df_unmonitored
    del df_monitored_sample
    gc.collect()

    classify(dataset, OUTPUT_ACC)


if __name__ == '__main__':
    ow_experiment()
