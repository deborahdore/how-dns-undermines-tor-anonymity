from os.path import join, abspath, dirname, pardir

# # # # # #  DIRECTORIES

# base dir
BASE_DIR = abspath(join(dirname(__file__), pardir, pardir, pardir))

# base -> src : sources directory
SOURCE_DIR = join(BASE_DIR, 'src')

# base -> datasets : dataset directory
DATASET_DIR = join(BASE_DIR, 'datasets')

# base -> src -> collection : directory with dataset collection files
COLLECTION_DIR = join(SOURCE_DIR, 'collection')
# base -> src -> classification : directory with model implementation files
CLASSIFICATION_DIR = join(SOURCE_DIR, 'classification')

# base -> src -> classification -> results : directory with the results of the evalution of the model
RESULTS_DIR = join(CLASSIFICATION_DIR, 'results')
# base -> src -> classification -> models : directory with the model
MODELS_DIR = join(CLASSIFICATION_DIR, 'models')

# base -> src -> classification -> results -> plot : directory containing the plot of the evaluation
PLOT_DIR = join(RESULTS_DIR, 'plot')

# # # # # #  FILES

# base -> src -> collection: short_list_1500. A file the 1500 most visited websites by Alexa
ALL_URL_LIST = join(COLLECTION_DIR, 'short_list_1500')

# base -> src -> dataset -> OW: directory with the ngrams of unmonitored websites
DATASET_OPEN_WORLD = join(DATASET_DIR, 'OW')
# base -> src -> dataset -> LOC1: directory with the ngrams of monitored websites
DATASET_CLOSED_WORLD = join(DATASET_DIR, "LOC1")

# base -> src -> classification -> models : file with the model
RANDOM_FOREST_FILE = join(MODELS_DIR, "random_forest.pkl")

# base -> src -> classification -> results: results of the evaluation
OUTPUT_REPORT = join(RESULTS_DIR, 'ow_results.txt')

# base -> src -> classification -> plot: roc_curve.svg + precision_recall_curve.svg
ROC_CURVE_FILE = join(PLOT_DIR, "roc_curve.svg")
PR_CURVE_FILE = join(PLOT_DIR, "precision_recall_curve.svg")
