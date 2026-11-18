# ECMM463
A reproduction of _Phishing URL Detection: A Network-based Approach Robust to Evasion_ by Kim et al. (2022). The paper proposes a network-based approach to phishing URL detection that is robust to evasion attacks.

The code for this project consists of five Jupyter notebooks and two supplementary library files. The notebooks are:
- `make_graph.ipynb`: Creates a network from a dataset of labelled good/bad URLs. This network is saved as a .gexf file for the later steps and visualisation in Gephi.
- `make_embeddings.ipynb`: Generates a file of embeddings for a given graph.
- `predict.ipynb`: Trains and evaluates the belief-propagation algorithm, implemented in the form of the Predictor class. This has methods to train the model and, once trained, to predict the label of a given URL. A graph and an embedding file are required.
- `baselines.ipynb`: Computes the evaluation metrics for the selected baseline models, random forest and Polonium.
- `evasion_experiments.ipynb`: Evaluates the three models with different evasion ratios, producing the graph shown in Figure 1.
The library file `predictor.py` contains the actual prediction implementation, and `url_graph.py` has an abstraction of a NetworkX graph specific to this domain.
