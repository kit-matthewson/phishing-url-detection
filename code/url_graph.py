import re
import urllib.parse

import networkx as nx
import nltk  # type: ignore[import-untyped]
from nltk.corpus import stopwords  # type: ignore[import-untyped]

nltk.download("stopwords")
STOP_WORDS = set(stopwords.words("english"))

class URLGraph:
    """
    A class to represent a graph of URLs, their hostnames, and segmented words.
    """

    def __init__(self):
        """Initialize an empty URLGraph."""
        self._graph = nx.Graph()


    @staticmethod
    def read_gexf(file_path: str):
        """
        Read a GEXF file and return a URLGraph instance.

        Args:
            file_path: The path to the GEXF file.

        Returns:
            A URLGraph instance.
        """

        graph = nx.read_gexf(file_path)

        url_graph = URLGraph()
        url_graph._graph = graph

        return url_graph

    def add_url(self, url: urllib.parse.ParseResult, malicious: bool | None = None):
        """
        Add a parsed URL to the graph, creating nodes for the URL, its hostname, and its segmented words.

        Args:
            url: The parsed URL to add to the graph.
            malicious: A boolean indicating if the URL is malicious (True), not malicious (False), or unknown (None).
        """
        node_name = url_to_node_name(url.geturl())
        self._graph.add_node(node_name, type="url", malicious=malicious)

        def add_node_with_edge_for_url(node_text: str, node_type: str):
            node_text = clean_text(node_text)

            if not self._graph.has_node(node_text):
                self._graph.add_node(node_text, type=node_type)

            self._graph.add_edge(node_name, node_text)

        # Add the hostname as a node if it doesn't exist
        if url.hostname:
            add_node_with_edge_for_url(url.hostname, "hostname")

        # Draw edges from the URL to each word it contains
        words = self._segment_url(url)

        for word in words:
            add_node_with_edge_for_url(word, "word")

    def number_of_nodes(self) -> int:
        """Return the number of nodes in the graph."""
        return self._graph.number_of_nodes()

    def number_of_edges(self) -> int:
        """Return the number of edges in the graph."""
        return self._graph.number_of_edges()

    def get_graph(self) -> nx.Graph:
        """Return the underlying NetworkX graph."""
        return self._graph

    def save(self, output_path: str):
        """Save the graph to a GEXF file."""
        nx.write_gexf(self._graph, output_path)

    def _segment_url(self, url: urllib.parse.ParseResult) -> set:
        """
        Split a parsed URL into unique, non-empty, non-stop-word segments.

        Hostname components, path components, and query parameter names and values are separated using specific delimiters.

        Args:
            url: Parsed URL to segment.

        Returns:
            A set of unique, non-empty, non-stop-word segments from the URL.
        """
        HOSTNAME_SPLIT = re.compile(r"[\.]") # Split on '.'
        PATH_SPLIT = re.compile(r"[\/.!&,#$%;&:]") # Split on '/', '.', '!', '&', ',', '#', '$', '%', ';', and ':'
        QUERY_SPLIT = re.compile(r"[=&]")

        words = set()
        if url.hostname:
            words.update(re.split(HOSTNAME_SPLIT, url.hostname))
        if url.path:
            words.update(re.split(PATH_SPLIT, url.path))
        if url.query:
            words.update(re.split(QUERY_SPLIT, url.query))

        # Filter out empty strings and stop words from the set of words
        words = {word for word in words if word}
        words = words.difference(STOP_WORDS)

        return words


def clean_text(value: object) -> str:
    """
    Clean a string by removing invalid XML characters and encoding it to ASCII.

    Args:
        value: The input value to clean, which will be converted to a string.

    Returns:
        A cleaned string.
    """
    text = str(value)

    # Invalid XML characters
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Encode to ASCII, ignoring non-ASCII characters
    text = text.encode("ascii", "ignore").decode("ascii")

    return text


def url_to_node_name(url: str) -> str:
    """
    Convert a URL string to a cleaned node name suitable for graph representation.

    Args:
        url: The input URL string to clean.
    Returns:
        A cleaned string suitable for use as a node name in the graph.
    """
    parsed_url = urllib.parse.urlparse(url)
    return clean_text(parsed_url.geturl())
