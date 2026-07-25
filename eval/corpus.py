"""
Fixed local corpus backing the benchmark's web_search tool.

Per the project's resolved search-backend decision, evals run against
this fixed local corpus rather than a live search API, so benchmark runs
are fully reproducible: no network flakiness, no non-determinism, no API
cost. Swap `search()`'s backend for a real API only if representativeness
starts to matter more than reproducibility.

Documents are short factual notes on GNN architectures and mechanistic
interpretability results — the domain the benchmark questions in
`eval/qa_dataset.py` draw on. Multi-hop questions are deliberately
answerable only by combining facts that live in *different* documents.
"""
from __future__ import annotations

DOCUMENTS: dict[str, str] = {
    "gcn": (
        "Graph Convolutional Networks (GCN), introduced by Kipf and Welling in 2017, "
        "perform semi-supervised classification on graphs by propagating and "
        "averaging neighbor features through a layer-wise first-order approximation "
        "of spectral graph convolutions."
    ),
    "graphsage": (
        "GraphSAGE, introduced by Hamilton, Ying, and Leskovec in 2017, is an "
        "inductive representation learning method: instead of learning a fixed "
        "embedding per node, it learns aggregator functions that sample and "
        "combine features from a node's local neighborhood, so it generalizes to "
        "unseen nodes at inference time."
    ),
    "gat": (
        "Graph Attention Networks (GAT), introduced by Velickovic et al. in 2018, "
        "replace GCN's fixed neighbor-averaging with learned attention "
        "coefficients, letting each node weight its neighbors' contributions "
        "differently rather than treating them uniformly."
    ),
    "mpnn": (
        "Gilmer et al. (2017) introduced Message Passing Neural Networks (MPNN), "
        "a unified framework casting most graph neural network variants as "
        "rounds of message passing between nodes followed by an update function. "
        "The paper's motivating application was predicting quantum chemical "
        "properties of molecules from their graph structure."
    ),
    "gin": (
        "The Graph Isomorphism Network (GIN), introduced by Xu et al. in 2019, "
        "analyzes GNN expressiveness through the message-passing framework and "
        "shows that a sum aggregator, unlike GCN's or GraphSAGE's aggregators, "
        "lets a GNN be as powerful as the Weisfeiler-Lehman graph isomorphism "
        "test at distinguishing non-isomorphic graphs."
    ),
    "hgnn": (
        "Hypergraph Neural Networks (HGNN), introduced by Feng et al. in 2019, "
        "extend graph convolution to hyperedges that can connect more than two "
        "nodes at once, using a hyperedge convolution operation to encode "
        "higher-order data correlations that pairwise graphs cannot represent."
    ),
    "zoom_in_circuits": (
        "Olah et al.'s 2020 Distill article 'Zoom In: An Introduction to "
        "Circuits' proposed that neural networks can be understood by finding "
        "circuits: meaningful algorithms implemented by weights connecting "
        "features across layers. Their central example was curve detector "
        "neurons in InceptionV1, identified via feature visualization."
    ),
    "transformer_circuits": (
        "Elhage et al.'s 2021 'A Mathematical Framework for Transformer "
        "Circuits' decomposed attention-only transformers into interpretable "
        "QK (query-key) and OV (output-value) circuits and virtual weights "
        "formed by composing attention heads across layers, extending the "
        "circuits agenda from 'Zoom In' to language transformers."
    ),
    "induction_heads": (
        "Olsson et al.'s 2022 paper 'In-context Learning and Induction Heads' "
        "used the QK/OV circuit framework from Elhage et al.'s transformer "
        "circuits paper to identify induction heads: attention heads that "
        "detect a repeated token sequence and copy what previously followed it, "
        "which the authors argue drives much of in-context learning."
    ),
    "toy_superposition": (
        "Elhage et al.'s 2022 'Toy Models of Superposition' proposed that "
        "neural networks represent more features than they have dimensions by "
        "storing them as non-orthogonal directions that interfere with each "
        "other, a phenomenon called superposition, and showed it is more "
        "pronounced when features are sparse."
    ),
    "monosemanticity_sae": (
        "Bricken et al.'s 2023 'Towards Monosemanticity' used sparse "
        "autoencoders, a dictionary-learning method, to decompose superposed, "
        "polysemantic neuron activations described in 'Toy Models of "
        "Superposition' into a larger set of sparse, monosemantic features."
    ),
    "grokking": (
        "Power et al.'s 2022 paper on grokking showed that small transformers "
        "trained on algorithmic tasks (such as modular arithmetic) can reach "
        "perfect validation accuracy long after they have already reached near "
        "perfect training accuracy, a delayed generalization effect."
    ),
    "causal_scrubbing": (
        "Chan, Garriga-Alonso, et al. (Redwood Research, 2022) introduced "
        "causal scrubbing, a technique that tests a hypothesized circuit by "
        "replacing activations with resampled ones along every path the "
        "hypothesis claims is unimportant, checking whether model behavior is "
        "preserved -- a generalization of the activation patching used to "
        "validate the induction-head hypothesis."
    ),
    "transformerlens": (
        "TransformerLens, an open-source library created by Neel Nanda, "
        "provides hooks into every internal activation of GPT-2-style "
        "transformers, making techniques from the transformer circuits and "
        "induction heads papers -- such as activation patching -- practical to "
        "run on real pretrained models rather than only toy models."
    ),
}


def _words(text: str) -> set[str]:
    return {w.strip(".,()'\"-:;") .lower() for w in text.split()}


def _score(query: str, text: str) -> int:
    query_terms = {t for t in _words(query) if len(t) > 2}
    text_terms = _words(text)
    return sum(1 for term in query_terms if term in text_terms)


def search(query: str, top_k: int = 3) -> str:
    """Keyword-overlap search over the fixed local corpus. Deterministic
    and dependency-free, matching the project's reproducibility-over-
    representativeness decision on the search backend."""
    ranked = sorted(DOCUMENTS.items(), key=lambda kv: _score(query, kv[1]), reverse=True)
    top = [text for _, text in ranked[:top_k] if _score(query, text) > 0]
    if not top:
        return "No relevant documents found."
    return "\n\n".join(top)
