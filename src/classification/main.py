from src.classification.eval import eval
from src.classification.train import train
from src.classification.utils.utility import *


def main():
    """
    It loads the dataset, splits it into training and test sets, and then trains and evaluates two different models (Random
    Forest and K Nearest Neighbor) using two different n-gram sizes (1 and 2)
    """
    X_train, X_test, y_train, y_test = load_split_dataset()

    max_length_ngram = [1, 2]

    for i in max_length_ngram:
        logger.info("Starting experiment using a Random Forest Classifier")
        train(X_train, y_train, "RF", ngrams=i)

        logger.info("Starting evaluation using a Random Forest Classifier")
        eval(X_train, X_test, y_train, y_test, "RF", ngrams=i)

        logger.info("Starting experiment using a K Nearest Neighbor")
        train(X_train, y_train, "KNN", ngrams=i)

        logger.info("Starting evaluation using a K Nearest Neighbor")
        eval(X_train, X_test, y_train, y_test, "KNN", ngrams=i)


if __name__ == '__main__':
    # start experiment
    main()
