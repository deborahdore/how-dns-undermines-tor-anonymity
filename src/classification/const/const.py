from os.path import join, abspath, dirname, pardir

# # # # # #  DIRECTORIES

BASE_DIR = abspath(join(dirname(__file__), pardir, pardir, pardir))
COLLECTION_DIR = join(BASE_DIR, 'src', 'collection')
MODULE_DIR = abspath(join(dirname(__file__), pardir))
DATASET_DIR = join(BASE_DIR, 'datasets')

RESULTS_DIR = join(MODULE_DIR, 'results')

MODELS_DIR = join(RESULTS_DIR, 'model')
PLOT_DIR = join(RESULTS_DIR, 'plot')
REPORT_DIR = join(RESULTS_DIR, 'report')

# # # # # #  FILES

ALL_URL_LIST = join(COLLECTION_DIR, 'short_list_1500')

DATASET_OPEN_WORLD = join(DATASET_DIR, 'OW')
DATASET_CLOSED_WORLD = join(DATASET_DIR, "CW")

RANDOM_FOREST_FILE = join(MODELS_DIR, "random_forest_ngrams_{ngrams}.pkl")
KNEIGHBORS_FILE = join(MODELS_DIR, "knearest_neighbors_ngrams_{ngrams}.pkl")

OUTPUT_REPORT = join(REPORT_DIR, 'ow_report_result_{name}_ngrams_{ngrams}.txt')

ROC_CURVE_FILE = join(PLOT_DIR, "roc_curve_{name}_ngrams_{ngrams}.svg")
PR_CURVE_FILE = join(PLOT_DIR, "precision_recall_curve_{name}_ngrams_{ngrams}.svg")
