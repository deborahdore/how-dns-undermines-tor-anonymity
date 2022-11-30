import time
import warnings

from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, FeatureUnion

from src.classification.ngrams_extractor import NgramsExtractor
from src.classification.const.const import RANDOM_FOREST_FILE, KNEIGHBORS_FILE
from src.classification.utils.utility import create_random_grid, save_model

warnings.filterwarnings("ignore")


def train(X_train, y_train, model_type, ngrams=1):
    """
    It creates a pipeline that first extracts the ngrams from the text, then uses a randomized search to find the best
    combination of parameters for the model

    :param X_train: The training data
    :param y_train: the training labels
    :param model_type: The type of model to train. Either "RF" for Random Forest or "KNN" for K-Nearest Neighbors
    :param ngrams: The number of ngrams to use, defaults to 1 (optional)
    """

    if model_type == "RF":
        model = RandomForestClassifier()
        path = RANDOM_FOREST_FILE.format(ngrams=ngrams)
    else:
        model = KNeighborsClassifier()
        path = KNEIGHBORS_FILE.format(ngrams=ngrams)

    random_grid = create_random_grid(model_type)

    search_cv = RandomizedSearchCV(estimator=model,
                                   param_distributions=random_grid,
                                   n_iter=50,
                                   cv=5,
                                   verbose=2,
                                   n_jobs=-1)

    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('ngrams', NgramsExtractor(max_ngram_len=ngrams)),
        ])),
        ('rs_cv', search_cv)
    ])

    start = time.time()
    logger.info("Searching for the best combination of parameters")

    pipeline.fit(X_train, y_train.ravel())
    logger.info(f"{model_type} training time with {ngrams} ngrams: {time.time() - start}")

    logger.info("Saving best model")
    save_model(pipeline['rs_cv'].best_estimator_, path)

    logger.info(f"Best params: {pipeline['rs_cv'].best_params_}")
