import networkx as nx
import numpy as np
from sklearn.metrics import (  # type: ignore
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import KFold  # type: ignore
from tqdm import tqdm  # type: ignore
from url_graph import URLGraph, url_to_node_name


class Predictor:
    """
    A class to predict the label of a URL based on its graph representation using loopy belief propagation.
    """

    N_FOLDS = 3
    MAX_EPOCHS = 3

    def __init__(
        self,
        graph: URLGraph,
        embeddings: dict[str, list[float]],
    ):
        """
        Initialize the Predictor with a URLGraph.

        Args:
            graph: An instance of URLGraph containing the URL nodes and their relationships.
            embeddings: Vectors keyed by graph node name. Euclidean distances
                between vectors are used as edge distances.
        """

        self._graph: nx.Graph = graph.get_graph()
        self._embeddings = embeddings

        # Get the set of nodes that have a known label (malicious or not malicious)
        self._has_ground_truth = np.array(
            [
                node
                for node, data in self._graph.nodes(data=True)
                if data.get("malicious") is not None
            ]
        )

    def train(self, output=True):
        kf = KFold(n_splits=self.N_FOLDS, shuffle=True, random_state=42)
        fold_metrics = []

        for train, test in kf.split(self._has_ground_truth):
            train_nodes = self._has_ground_truth[train]
            test_nodes = self._has_ground_truth[test]

            num_malicious = 0
            num_benign = 0

            train_node_set = set(train_nodes)

            # Set node labels
            for node, data in self._graph.nodes(data=True):
                data.update(
                    {
                        "label": None,
                        "predicted_label": None,
                        "best_label": -1,
                        "msgbox": {
                            neighbor: [0, 0] for neighbor in self._graph.neighbors(node)
                        },
                        "msg_comp": [0, 0],
                    }
                )

                if node not in train_node_set:
                    data["cost"] = [0.5, 0.5]
                    continue

                is_malicious = data.get("malicious")

                if is_malicious is True:
                    data["label"] = 1
                    data["cost"] = [0.99, 0.01]
                    num_malicious += 1
                elif is_malicious is False:
                    data["label"] = 0
                    data["cost"] = [0.01, 0.99]
                    num_benign += 1
                else:
                    raise ValueError(
                        f"Training Node {node!r} has a label that is neither True nor False."
                    )

            self._set_edge_sims()

            # Perform belief propogation
            if output:
                epoch_iter = tqdm(range(self.MAX_EPOCHS), desc="Epochs", unit="epoch")
            else:
                epoch_iter = range(self.MAX_EPOCHS)
            
            for epoch in epoch_iter:
                self._propogate_messages()
                self._set_predictions()

            fold_metrics.append(self._metrics(test_nodes))
            
            if output:
                self._print_fold_performance(len(fold_metrics), fold_metrics[-1])
                print()

        averages = np.mean(fold_metrics, axis=0)

        if output:
            print(
                "Cross-validation averages: "
                f"precision={averages[0]:.3f}, recall={averages[1]:.3f}, "
                f"f1={averages[2]:.3f}, accuracy={averages[3]:.3f}"
            )

    def predict(self, url: str) -> int:
        """
        Predict the label of a given URL.

        Args:
            url: The URL to predict.

        Returns:
            The predicted label (0 for benign, 1 for malicious).
        """
        node_name = url_to_node_name(url)

        if not self._graph.has_node(node_name):
            raise ValueError(f"URL '{url}' not found in the graph.")

        return int(self._graph.nodes[node_name].get("predicted_label", -1))

    def _propogate_messages(self):
        """
        Propagate messages between nodes in the graph based on their current beliefs and the structure of the graph.
        This method updates the 'msgbox' and 'msg_comp' attributes of each node based on the messages received from its neighbors.
        """
        graph = self._graph

        for node in graph.nodes():
            if graph.nodes[node]["label"] != None:
                for nbr in graph.neighbors(node):
                    if graph.nodes[nbr]["label"] == None:
                        self._send_msg_label(node, nbr)

        for node in graph.nodes():
            if graph.nodes[node]["label"] == None:
                for nbr in graph.neighbors(node):
                    if graph.nodes[nbr]["label"] == None:
                        self._send_msg(node, nbr)

    def _set_edge_sims(self):
        for sender, receiver in self._graph.edges():
            try:
                sender_embedding = np.asarray(self._embeddings[sender], dtype=float)
                receiver_embedding = np.asarray(self._embeddings[receiver], dtype=float)
            except KeyError as error:
                raise ValueError(f"Embedding is missing for node {error.args[0]!r}.")

            # Use cosine similarity to set the similarity attribute for the edge
            dot_product = np.dot(sender_embedding, receiver_embedding)
            norm_sender = np.linalg.norm(sender_embedding)
            norm_receiver = np.linalg.norm(receiver_embedding)

            if norm_sender == 0 or norm_receiver == 0:
                similarity = 0.0
            else:
                similarity = float(dot_product / (norm_sender * norm_receiver))

            self._graph.edges[sender, receiver]["sim"] = similarity

    def _set_predictions(self):
        for _, data in self._graph.nodes(data=True):
            cost_not_malicious = data["cost"][0] + data["msg_comp"][0]
            cost_malicious = data["cost"][1] + data["msg_comp"][1]

            predicted_label = int(cost_malicious < cost_not_malicious)

            data["predicted_label"] = predicted_label
            data["best_label"] = predicted_label

    def _send_msg_label(self, sender, receiver):
        """
        Send a message from a labeled sender node to an unlabeled receiver node.
        The message is based on the sender's label and the edge potential between the sender and receiver.

        Args:
            sender: The node sending the message (must have a known label).
            receiver: The node receiving the message (must not have a known label).
        """
        receiver_data = self._graph.nodes[receiver]

        # Observed nodes send their known label directly, as in the reference
        # implementation. Edge compatibility is applied to hidden nodes.
        msg = [1, 0] if self._graph.nodes[sender]["label"] == 1 else [0, 1]

        previous_msg = receiver_data["msgbox"][sender]

        receiver_data["msg_comp"][0] += msg[0] - previous_msg[0]
        receiver_data["msg_comp"][1] += msg[1] - previous_msg[1]
        receiver_data["msgbox"][sender] = msg

    def _send_msg(self, sender, receiver):
        """
        Send a message from an unlabeled sender node to an unlabeled receiver node.
        The message is based on the sender's current beliefs and the edge potential between the sender and receiver.

        Args:
            sender: The node sending the message (must not have a known label).
            receiver: The node receiving the message (must not have a known label).
        """
        msg = self._min_sum(sender, receiver)

        receiver_data = self._graph.nodes[receiver]
        previous_msg = receiver_data["msgbox"][sender]

        receiver_data["msg_comp"][0] += msg[0] - previous_msg[0]
        receiver_data["msg_comp"][1] += msg[1] - previous_msg[1]
        receiver_data["msgbox"][sender] = msg

    def _min_sum(self, sender, receiver, compat_threshold1=0.7, compat_threshold2=0.7):
        """
        Compute the minimum sum message from the sender to the receiver based on the sender's current beliefs and the edge potential.

        Args:
            sender: The node sending the message.
            receiver: The node receiving the message.
        Returns:
            A list containing the computed message for both labels (malicious and benign).
        """
        sender_data = self._graph.nodes[sender]

        new_msg = [0.0, 0.0]

        for i in range(2):
            p_not_related = 0.0
            p_related = 0.0

            p_not_related += sender_data["cost"][0]
            p_related += sender_data["cost"][1]

            p_not_related += (
                sender_data["msg_comp"][0] - sender_data["msgbox"][receiver][0]
            )
            p_related += sender_data["msg_comp"][1] - sender_data["msgbox"][receiver][1]

            # Use the 'table3' smoothing method
            p_not_related += (
                np.min(
                    [compat_threshold1, 1 - self._graph.edges[sender, receiver]["sim"]]
                )
                if i == 0
                else np.max(
                    [compat_threshold2, self._graph.edges[sender, receiver]["sim"]]
                )
            )
            p_related += (
                np.max([compat_threshold2, self._graph.edges[sender, receiver]["sim"]])
                if i == 0
                else np.min(
                    [compat_threshold1, 1 - self._graph.edges[sender, receiver]["sim"]]
                )
            )

            new_msg[i] = min(p_not_related, p_related)

        # Normalise the message to prevent numerical issues
        new_msg = [m - min(new_msg) for m in new_msg]

        return new_msg

    def _metrics(self, nodes) -> tuple[float, float, float, float]:
        labelled = [
            node
            for node in nodes
            if self._graph.nodes[node].get("malicious") is not None
        ]
        
        if not labelled:
            return 0.0, 0.0, 0.0, 0.0

        actual = [int(self._graph.nodes[node]["malicious"]) for node in labelled]
        predicted = [self._prediction(node) for node in labelled]

        # Calculate precision, recall, f1-score, and accuracy using sklearn metrics
        precision = precision_score(actual, predicted, zero_division=0)
        recall = recall_score(actual, predicted, zero_division=0)
        f1 = f1_score(actual, predicted, zero_division=0)
        accuracy = accuracy_score(actual, predicted)
        
        return float(precision), float(recall), float(f1), float(accuracy)

    def _prediction(self, node) -> int:
        predicted = self._graph.nodes[node].get("predicted_label")
        if predicted not in (0, 1):
            raise RuntimeError(
                f"Node {node!r} has an invalid predicted label: {predicted!r}. "
                "Expected 0 or 1."
            )
        return int(predicted)

    def _print_fold_performance(
        self, fold: int, metrics: tuple[float, float, float, float]
    ):
        print(
            f"Fold {fold}: precision={metrics[0]:.3f}, "
            f"recall={metrics[1]:.3f}, f1={metrics[2]:.3f}, "
            f"accuracy={metrics[3]:.3f}"
        )
