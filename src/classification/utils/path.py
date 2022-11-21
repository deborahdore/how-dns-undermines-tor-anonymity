from os.path import join, abspath, dirname, pardir

# # # # # #  DIRECTORIES

BASE_DIR = abspath(join(dirname(__file__), pardir, pardir, pardir))
SOURCE_DIR = join(BASE_DIR, 'src')

DATASET_DIR = join(BASE_DIR, 'datasets')

COLLECTION_DIR = join(SOURCE_DIR, 'collection')
CLASSIFICATION_DIR = join(SOURCE_DIR, 'classification')

RESULTS_DIR = join(CLASSIFICATION_DIR, 'results')
MODELS_DIR = join(CLASSIFICATION_DIR, 'models')

PLOT_DIR = join(RESULTS_DIR, 'plot')

# # # # # #  FILES

ALL_URL_LIST = join(COLLECTION_DIR, 'short_list_1500')

DATASET_OPEN_WORLD = join(DATASET_DIR, 'OW')
DATASET_CLOSED_WORLD = join(DATASET_DIR, "CW")

RANDOM_FOREST_FILE = join(MODELS_DIR, "random_forest.pkl")

OUTPUT_REPORT = join(RESULTS_DIR, 'ow_report_result.txt')
OUTPUT_PRT = join(RESULTS_DIR, 'ow_prt_results.csv')

ROC_CURVE_FILE = join(PLOT_DIR, "roc_curve.svg")
PR_CURVE_FILE = join(PLOT_DIR, "precision_recall_curve.svg")
