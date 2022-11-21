# Classification

This directory contains the code used to implement and evaluate a classifier whose job is to distinguish between
monitored and unmonitored websites. The classifier uses as a feature the TLS length of packets in a trace. The
experiment can be initiated by running the [main](main.py) script contained in this folder. It will first search for the
best parameter of a Random Forest Classifier, and then evaluate the results. <br>
After the completion of the script, the resulting reports can be found in the [results folder](results) along with
the [plots](results/plot).

- The dataset can be found [here](../../datasets).
- The list of monitored websites used can be found [here](../collection/short_list_1500).
- An already trained model can be found [here](models)

## Results

Results display a good ability of the classifier to distinguish between classes. More details in
the [report](/reports/Assignment%232.pdf).

![plot](results/plot/roc_curve.svg)