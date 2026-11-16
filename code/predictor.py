import networkx as nx
import numpy as np
from sklearn.model_selection import KFold  # type: ignore
from tqdm import tqdm  # type: ignore
from url_graph import URLGraph, url_to_node_name


class Predictor:
    """
    A class to predict the label of a URL based on its graph representation using loopy belief propagation.
    """

    N_FOLDS = 5
    MAX_EPOCHS = 5

    def __init__(
        self,
        graph: URLGraph,
        embeddings: dict[str, list[float]] | None = None,
        compatibility_type: str = "distance",
    ):
        """
        Initialize the Predictor with a URLGraph.

        Args:
            graph: An instance of URLGraph containing the URL nodes and their relationships.
            embeddings: Optional vectors keyed by graph node name. Euclidean distances
                between vectors are used as edge distances.
            compatibility_type: The message compatibility rule, either ``"distance"``
                or ``"polonium"``.
        """

        self._graph: nx.Graph = graph.get_graph()
        self._embeddings = embeddings
        self._compatibility_type = compatibility_type

        if compatibility_type not in {"polonium", "distance"}:
            raise ValueError(
                f"Unknown compatibility type: {compatibility_type!r}. Expected 'polonium' or 'distance'."
            )

        # Get the set of nodes that have a known label (malicious or not malicious)
        self._has_ground_truth = np.array(
            [
                node
                for node, data in self._graph.nodes(data=True)
                if data.get("malicious") is not None
            ]
        )

    def train(self):
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

            self._set_edge_potentials()

            # Perform belief propogation
            for epoch in tqdm(range(self.MAX_EPOCHS), desc="Epochs", unit="epoch"):
                self._propogate_messages()
                self._set_predictions()

            fold_metrics.append(self._metrics(test_nodes))
            self._print_fold_performance(len(fold_metrics), fold_metrics[-1])
            print()

        averages = np.mean(fold_metrics, axis=0)
        
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

    def _set_edge_potentials(self):
        if self._embeddings is None:
            for edge in self._graph.edges():
                self._graph.edges[edge]["distance"] = 1.0
                self._graph.edges[edge]["sim"] = 0.5
            return

        edge_distances = {}

        for sender, receiver in self._graph.edges():
            try:
                sender_embedding = np.asarray(self._embeddings[sender], dtype=float)
                receiver_embedding = np.asarray(
                    self._embeddings[receiver], dtype=float
                )
            except KeyError as error:
                raise ValueError(
                    f"Embedding is missing for graph node {error.args[0]!r}."
                ) from error

            if (
                sender_embedding.ndim != 1
                or receiver_embedding.ndim != 1
                or sender_embedding.shape != receiver_embedding.shape
            ):
                raise ValueError(
                    f"Embedding dimensions do not match for edge "
                    f"{sender!r}, {receiver!r}."
                )

            distance = float(np.linalg.norm(sender_embedding - receiver_embedding))
            if not np.isfinite(distance):
                raise ValueError(
                    f"Embedding distance is not finite for edge "
                    f"{sender!r}, {receiver!r}."
                )
            edge_distances[(sender, receiver)] = distance

        distances = np.array(list(edge_distances.values()))
        min_distance = float(distances.min())
        distance_range = float(distances.max() - min_distance)

        for edge, distance in edge_distances.items():
            normalised_distance = (
                0.5
                if distance_range == 0
                else (distance - min_distance) / distance_range
            )
            self._graph.edges[edge]["distance"] = normalised_distance
            self._graph.edges[edge]["sim"] = 1.0 - normalised_distance

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

    def _min_sum(self, sender, receiver):
        """
        Compute the minimum sum message from the sender to the receiver based on the sender's current beliefs and the edge potential.

        Args:
            sender: The node sending the message.
            receiver: The node receiving the message.
        Returns:
            A list containing the computed message for both labels (malicious and benign).
        """
        EPSILON = 1e-10

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

            match self._compatibility_type:
                case "polonium":
                    if i == 0:
                        p_not_related += 0.5 - EPSILON
                        p_related += 0.5 + EPSILON
                    else:
                        p_not_related += 0.5 + EPSILON
                        p_related += 0.5 - EPSILON
                case "distance":
                    p_not_related += (
                        0 if i == 0 else self._graph.edges[sender, receiver]["distance"]
                    )
                    p_related += (
                        self._graph.edges[sender, receiver]["distance"] if i == 0 else 0
                    )
                case _:
                    raise ValueError(
                        f"Unknown compatibility type: {self._compatibility_type}"
                    )

            new_msg[i] = min(p_not_related, p_related)

        # Normalise the message to prevent numerical issues
        return [m - min(new_msg) for m in new_msg]

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

        true_positive = sum(
            target == 1 and guess == 1 for target, guess in zip(actual, predicted)
        )
        false_positive = sum(
            target == 0 and guess == 1 for target, guess in zip(actual, predicted)
        )
        false_negative = sum(
            target == 1 and guess == 0 for target, guess in zip(actual, predicted)
        )

        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = (
            (2 * precision * recall / (precision + recall))
            if precision + recall
            else 0.0
        )
        accuracy = sum(
            target == guess for target, guess in zip(actual, predicted)
        ) / len(labelled)
        return precision, recall, f1, accuracy

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
