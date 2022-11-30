from src.classification.eval import eval
from src.classification.train import train
from src.classification.utils.utility import *


def main():
    X_train, X_test, y_train, y_test = load_split_dataset()

    ngrams = 2

    # logger.info("Starting experiment using a Random Forest Classifier")
    # train(X_train, y_train, "RF", ngrams)
    #
    # logger.info("Starting evaluation using a Random Forest Classifier")
    # eval(X_train, X_test, y_train, y_test, "RF", ngrams)

    logger.info("Starting experiment using a K Nearest Neighbor")
    train(X_train, y_train, "KNN", ngrams)

    logger.info("Starting evaluation using a K Nearest Neighbor")
    eval(X_train, X_test, y_train, y_test, "KNN", ngrams)


if __name__ == '__main__':
    main()
