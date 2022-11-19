# how-DNS-undermines-Tor-anonymity

The goal of the repository is to reproduce the model of the paper by Sandra Siby, Marc Juarez, Claudia Diaz,
Narseo Vallina-Rodriguez, Carmela Troncoso: "Encrypted DNS --> Privacy? A Traffic Analysis Perspective" and tune it to
obtain better attacks result.

The repository is organized as follows:

- [collection](src/collection): where the code and the instruction for the dataset collection are stored.
    - The code for the dataset collection is taken from [this](https://github.com/spring-epfl/doh_traffic_analysis) repo
      with modification due to compatibility with our systems.
- [classification](src/classification): contains the code for the classifier implementation and evaluation.
    - [results](src/classification/results): contains the results of the classifier evaluation
    - [models](src/classification/models): contains the trained model
- Other:
    - [dataset](datasets): contains the dataset used to train the model. Since the aquistion alone required too much
      time, the dataset is integrated with
      the *[open-world dataset](https://github.com/spring-epfl/doh_traffic_analysis/tree/master/dataset/OW])*
      and *[closed-world dataset LOC1](https://github.com/spring-epfl/doh_traffic_analysis/tree/master/dataset/LOC1])*
      previously collected by the authors of the paper.
    - [paper](paper): contains the report for the project course

*To run the whole experiment do*:

1. [collection](/src/collection)
2. [extraction](/src/extraction)
2. [classification](/src/classification)