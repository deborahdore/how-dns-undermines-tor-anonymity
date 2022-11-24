import json
import os
import re

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

from src.classification.utils.path import ALL_URL_LIST, DATASET_CLOSED_WORLD, DATASET_OPEN_WORLD

PATH_REGEX = {'name': r'(?P<name>\w+)',
              'dev': r'(?:(?P<dev>[^_]+)_)?',
              'sites': r'(?:(?P<sites>[^_]+)_)?',
              'date': r'(?P<date>\d\d-\d\d-\d\d)',
              'inst': r'(?:_(?P<inst>\d+))?'}
FNAME_REGEX = re.compile('{name}/{dev}{sites}{date}{inst}'.format(**PATH_REGEX))


def recover_order(sent_lengths, received_lengths, order):
    """Return sequence of lengths from snd/rcv lengths and order.

    Example:
        sent = [20, 33, 40]
        received = [33, 20, 20]
        order = [1, -1, 1, 1, -1, -1]
        Returns: [20, -33, 33, 40, -20, -20]
    """
    sequence = np.zeros(len(order))
    sequence[np.argwhere(order > 0).flatten()] = sent_lengths
    sequence[np.argwhere(order < 0).flatten()] = np.negative(received_lengths)
    return sequence


def get_bursts(len_seq):
    """Returns the sequence split by bursts.

    Example:
        len_seq = [20, -33, 33, 40, -20, -20]
        Returns: [[20], [-33], [33, 40], [-20, -20]]
    """
    directions = len_seq / abs(len_seq)
    index_dir_change = np.where(directions[1:] - directions[:-1] != 0)[0] + 1
    bursts = np.split(len_seq, index_dir_change)
    return bursts


def join_str(lengths):
    return ' '.join(map(str, lengths))


def it_webpages(fpath):
    """Iterate over all the websites contained in a file."""
    with open(fpath) as f:
        data_dict = json.loads(f.read())
        try:
            for pcap_filename, values in data_dict.items():
                webpage_num = pcap_filename[:-5]
                snd, rcv = values['sent'], values['received']
                order = values['order']
                lengths = recover_order(*map(np.array, [snd, rcv, order]))
                yield webpage_num, lengths
        except KeyError:
            logger.info(f"{fpath}does not have a known order sequence")
            return
        except Exception as e:
            logger.error(f"{fpath}, {pcap_filename}, {e}")


def sel_files(dpath):
    """Yield files that satisfy conditions."""
    sel_files = []
    for root, _, files in os.walk(dpath):
        for fname in files:
            if not fname.endswith('.json'):  # skip non-json files
                continue
            fpath = os.path.join(root, fname)
            sel_files.append(fpath)
    return sel_files


def load_data(dpath):
    """Traverse the directory and parse all the captures in it.

    Returns a dataframe containing encoded lengths.
    """
    logger.info("Starting to parse")
    selected_files = sel_files(dpath)
    logger.info(f"Number of selected files: {len(selected_files)}")

    # iterave over selected files and build dataframe
    empties = 0
    idx = pd.DataFrame(columns=PATH_REGEX.keys())
    for fpath in selected_files:
        m = FNAME_REGEX.search(fpath)
        if m is None:
            logger.error(f"{fpath}, {FNAME_REGEX.pattern}")
            continue
        row_head = {k: m.group(k) for k in PATH_REGEX}
        for i, (webpage_id, lengths) in enumerate(it_webpages(fpath)):
            if len(lengths) == 0:
                empties += 1
                continue
            row_head['fname'] = os.path.basename(fpath)
            row_head['class_label'] = webpage_id
            row_head['lengths'] = lengths
            idx = idx.append(row_head, ignore_index=True)
        logger.info(f'{i} sites in {fpath}')
    logger.info(f"Empty traces: {empties}")

    # fix some naming issues:
    idx['inst'] = idx.inst.fillna(0)
    idx['date'] = pd.to_datetime(idx.date.str.replace('-18', '-2018'),
                                 format='%d-%m-%Y')
    return idx


def load_mapping():
    """Return Alexa as a list."""
    return [l.strip() for l in open(ALL_URL_LIST)]


ALEXA_MAP = load_mapping()


def save_model(model, name):
    """
    **The function takes in a model, serializes it, and saves it to a file.**
    :param model: the model to save
    :param name: name of the file in which the model will be saved
    """
    joblib.dump(model, name)


def load_split_dataset():
    """
    It loads the two datasets, samples the monitored dataset to match the size of the unmonitored dataset, and then splits
    the data into training and testing sets
    :return: X_train, X_test, y_train, y_test
    """
    logger.info("Load Dataset")

    # open world data
    df_unmonitored = load_data(DATASET_OPEN_WORLD)
    df_unmonitored['target'] = "unmonitored"

    # closed world data
    df_monitored = load_data(DATASET_CLOSED_WORLD)
    df_monitored['target'] = "monitored"

    logger.info(f'Size monitored dataset: {df_monitored.shape[0]}')
    logger.info(f'Size unmonitored dataset: {df_unmonitored.shape[0]}')

    df_monitored_sample = df_monitored.sample(df_unmonitored.shape[0])

    dataset = pd.concat([df_monitored_sample, df_unmonitored], axis=0)

    logger.info(f'Dataset length: {dataset.shape[0]}')

    X = dataset[['lengths']]

    y = dataset[['target']]
    y = label_binarize(y, classes=['monitored', 'unmonitored'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True)

    return X_train, X_test, y_train, y_test


def load_model(path):
    """
    It loads the model from the file `RANDOM_FOREST_FILE` and returns it
    :return: The model is being returned.
    """
    return joblib.load(path)


def create_random_grid_knn():
    """
    It creates a dictionary of parameters for the KNN classifier
    :return: A dictionary of parameters for the KNN classifier.
    """
    grid_params = {'n_neighbors': [5, 7, 9, 11, 13, 15],
                   'weights': ['uniform', 'distance'],
                   'metric': ['minkowski', 'euclidean', 'manhattan'],
                   'algorithm': ['ball_tree', 'kd_tree', 'auto']}
    return grid_params


def create_random_grid(model_type):
    """
    If the model type is RF, return the random grid for RF. Otherwise, return the random grid for KNN

    :param model_type: The type of model to use. Either "RF" or "KNN"
    :return: A dictionary of parameters and their values
    """
    if model_type == "RF":
        return create_random_grid_rf()
    else:
        return create_random_grid_knn()


def create_random_grid_rf():
    """
    It creates a dictionary of hyperparameters to be used in a random search
    :return: A dictionary of the parameters to be used in the random search.
    """
    # num of trees
    n_estimators = [int(x) for x in np.linspace(start=200, stop=1000, num=10)]
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
    return random_grid
