# ECMM463

- [Coursework Description](https://ele.exeter.ac.uk/pluginfile.php/6088019/mod_resource/content/1/Coursework%20Description.pdf)
- [Paper](https://doi.org/10.1145/3548606.3560615)
- [Trello](https://trello.com/b/NjkFY0YJ/building-systems-coursework)

## Code
In the `notebooks/` directory you will find three notebooks:
1. `make_graph.ipynb` constructs a graph using `networkx`, where nodes are entities such as URLs, domains, IP addresses, and the words that construct these. The resulting graph is saved as a `.gexf` file for use in the next notebooks.
3. `make_embeddings.ipynb` reads a graph file and generates node embeddings for it using node2vec. The result is pickled and saved.
2. `predict.ipynb` reads a graph file and node embeddings from the previous notebooks, and validates the described prediction model on them.