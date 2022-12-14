# How DNS undermines Tor anonymity

The goal of the repository is to reproduce the model of the paper by Sandra Siby, Marc Juarez, Claudia Diaz,
Narseo Vallina-Rodriguez, Carmela Troncoso: _Encrypted DNS --> Privacy? A Traffic Analysis Perspective_ and tune it to
obtain better attacks result.

----- 
The repository is organized as follows:

``` bash
├── datasets
│        ├── CW
│        └── OW
├── paper
└── src
    ├── classification
    │        ├── const
    │        ├── results
    │        │       ├── model
    │        │       ├── plot
    │        │       └── report
    │        └── utils
    ├── collection
    └── extraction
        ├── result
        └── pcap

```

- [dataset](datasets): contains the dataset used to train the model. Since the acquisition alone required too much
  time, the dataset is integrated with
  the *[open-world dataset](https://github.com/spring-epfl/doh_traffic_analysis/tree/master/dataset/OW)*
  and *[closed-world dataset](https://github.com/spring-epfl/doh_traffic_analysis/tree/master/dataset/LOC1)*
  previously collected by the authors of the paper.
- [paper](paper): contains the reports for the project course
- [src](src): contains the sources
    - [collection](src/collection): where the code and the instruction for the dataset collection are stored.
        - The code for the dataset collection is taken from [this](https://github.com/spring-epfl/doh_traffic_analysis)
          repo with modification due to compatibility with our systems.
    - [extraction](src/extraction): contains the scripts for the extraction of the TLS record length in a trace for the
      PCAP files collected previously
        - [pcap](src/extraction/pcap): contains the pcap files to be analyzed
        - [result](src/extraction/result): contains the resulting dataset from the files analyzed
    - [classification](src/classification): contains the code for the classifier implementation and evaluation.
        - [results](src/classification/results): contains the results of the classifier evaluation
            - [model](src/classification/results/model): contains the trained model
            - [plot](src/classification/results/plot): contains the ROC curve and the PR curve for each classifier
            - [report](src/classification/results/report): contains reports on the accuracy, precision, recall and
              f1-score for each classifier

-------

*To run the whole experiment do*:

1. [collection](/src/collection)
2. [extraction](/src/extraction)
3. [classification](/src/classification)

Each directory contains more specific information in their Readme.md file.

--------

*Note:*
We suggest using anaconda and creating a new environment using the [environment](environment.yml) file to install the
correct packages.
