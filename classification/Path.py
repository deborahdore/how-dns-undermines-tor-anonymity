from os.path import join

BASE_DIR = "/Users/deborah/Documents/how-DNS-undermines-Tor-anonymity"
DATASET_DIR = join(BASE_DIR, 'datasets')
COLLEC_DIR = join(BASE_DIR, 'collection')
RESULTS_DIR = join(BASE_DIR, 'results')

DATASET_OPEN_WORLD = join(DATASET_DIR, 'OW')
DATASET_CLOSED_WORLD = join(DATASET_DIR, "LOC1")

ALL_URL_LIST = join(COLLEC_DIR, 'short_list_1500')

# Output files
OUTPUT_PROB = join(RESULTS_DIR, 'ow_result_prob')  # predicition probability for each class along with truth label
OUTPUT_ACC = join(RESULTS_DIR, 'ow_result_acc')  # accuracy for each fold
OUTPUT_STATISTICS = join(RESULTS_DIR, 'ow_stats')  # precision/recall/f-score stats (mean and std)
OUTPUT_REPORT = join(RESULTS_DIR, 'ow_report')  # detailed report (precision/recall/f-score for each class in each fold)
OUTPUT_TP = join(RESULTS_DIR, 'ow_tp')  # true vs predicted label (for other analysis if required)
