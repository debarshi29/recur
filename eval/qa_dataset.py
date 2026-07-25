"""
Multi-hop QA benchmark v0.

Domain: GNN architectures and mechanistic interpretability results (see
`eval/corpus.py` for the fixed local corpus these questions are answerable
from). Questions tagged `expected_hops >= 2` are deliberately unanswerable
from a single document in the corpus -- they require combining a fact from
one document with a fact from another, which is what actually
differentiates ReAct/Reflection/Plan-Execute on multi-hop reasoning rather
than on single-lookup recall.
"""
from __future__ import annotations

from harness.contracts import Task

TASKS: list[Task] = [
    Task(
        id="q01",
        question="Who introduced Graph Attention Networks, and in what year?",
        gold_answer="Velickovic et al., 2018",
        expected_hops=1,
        metadata={"difficulty": "easy", "topic": "gnn"},
    ),
    Task(
        id="q02",
        question="What technique did Bricken et al.'s 'Towards Monosemanticity' use to decompose superposed neuron activations?",
        gold_answer="sparse autoencoders",
        expected_hops=1,
        metadata={"difficulty": "easy", "topic": "interp"},
    ),
    Task(
        id="q03",
        question=(
            "The message-passing framework that GIN's theoretical analysis relies on "
            "was introduced in a paper motivated by which scientific domain?"
        ),
        gold_answer="quantum chemistry",
        expected_hops=2,
        metadata={"difficulty": "medium", "topic": "gnn"},
    ),
    Task(
        id="q04",
        question=(
            "What year was the paper published that first cast GNN variants as "
            "message passing, the same framework GIN uses to compare its "
            "expressiveness to the Weisfeiler-Lehman test?"
        ),
        gold_answer="2017",
        expected_hops=2,
        metadata={"difficulty": "medium", "topic": "gnn"},
    ),
    Task(
        id="q05",
        question=(
            "Which aggregator distinguishes GIN from GCN and GraphSAGE in matching "
            "the Weisfeiler-Lehman test's power to distinguish non-isomorphic graphs?"
        ),
        gold_answer="sum aggregator",
        expected_hops=2,
        metadata={"difficulty": "medium", "topic": "gnn"},
    ),
    Task(
        id="q06",
        question=(
            "GAT replaces a fixed neighbor-averaging scheme with learned "
            "attention coefficients. Which earlier architecture used that fixed "
            "averaging scheme, and what year was it introduced?"
        ),
        gold_answer="GCN, 2017",
        expected_hops=2,
        metadata={"difficulty": "medium", "topic": "gnn"},
    ),
    Task(
        id="q07",
        question=(
            "Hypergraph Neural Networks extend graph convolution beyond pairwise "
            "edges. What operation do they use to encode higher-order "
            "correlations, and who introduced it?"
        ),
        gold_answer="hyperedge convolution, Feng et al.",
        expected_hops=2,
        metadata={"difficulty": "medium", "topic": "gnn"},
    ),
    Task(
        id="q08",
        question=(
            "What central worked example did the Distill circuits article that "
            "originated the 'circuits' agenda use to demonstrate finding an "
            "algorithm implemented by network weights?"
        ),
        gold_answer="curve detector neurons in InceptionV1",
        expected_hops=1,
        metadata={"difficulty": "easy", "topic": "interp"},
    ),
    Task(
        id="q09",
        question=(
            "The induction heads paper identified attention heads that copy what "
            "previously followed a repeated token. Which earlier paper supplied "
            "the QK/OV circuit framework the induction heads paper used to find them?"
        ),
        gold_answer="A Mathematical Framework for Transformer Circuits",
        expected_hops=2,
        metadata={"difficulty": "medium", "topic": "interp"},
    ),
    Task(
        id="q10",
        question=(
            "What earlier, non-language-model work does Elhage et al.'s "
            "transformer circuits paper extend the circuits agenda from?"
        ),
        gold_answer="Zoom In: An Introduction to Circuits",
        expected_hops=2,
        metadata={"difficulty": "medium", "topic": "interp"},
    ),
    Task(
        id="q11",
        question=(
            "Causal scrubbing generalizes an earlier technique used to validate "
            "the induction-head hypothesis. Name that earlier technique."
        ),
        gold_answer="activation patching",
        expected_hops=2,
        metadata={"difficulty": "medium", "topic": "interp"},
    ),
    Task(
        id="q12",
        question=(
            "Which open-source library made activation patching -- the technique "
            "used to validate the induction-head hypothesis -- practical to run "
            "on real pretrained GPT-2-style models rather than only toy models, "
            "and who created it?"
        ),
        gold_answer="TransformerLens, Neel Nanda",
        expected_hops=3,
        metadata={"difficulty": "hard", "topic": "interp"},
    ),
    Task(
        id="q13",
        question=(
            "'Towards Monosemanticity' decomposes a phenomenon first named and "
            "studied in a 2022 paper. Name that phenomenon and the paper's title."
        ),
        gold_answer="superposition, Toy Models of Superposition",
        expected_hops=2,
        metadata={"difficulty": "medium", "topic": "interp"},
    ),
    Task(
        id="q14",
        question=(
            "According to Elhage et al., superposition is more pronounced under "
            "which condition on the features a network represents?"
        ),
        gold_answer="when features are sparse",
        expected_hops=1,
        metadata={"difficulty": "easy", "topic": "interp"},
    ),
    Task(
        id="q15",
        question=(
            "Grokking research on algorithmic tasks like modular arithmetic "
            "showed perfect validation accuracy arriving long after what other "
            "milestone was already reached?"
        ),
        gold_answer="near perfect training accuracy",
        expected_hops=1,
        metadata={"difficulty": "easy", "topic": "interp"},
    ),
    Task(
        id="q16",
        question=(
            "Between the paper that introduced inductive, sampling-based node "
            "aggregation for unseen nodes and the paper that introduced learned "
            "attention over neighbors, which was published first, and in what year?"
        ),
        gold_answer="GraphSAGE, 2017",
        expected_hops=2,
        metadata={"difficulty": "medium", "topic": "gnn"},
    ),
    Task(
        id="q17",
        question=(
            "What group first introduced the mathematical framework decomposing "
            "attention-only transformers into QK and OV circuits, and in what year?"
        ),
        gold_answer="Elhage et al., 2021",
        expected_hops=1,
        metadata={"difficulty": "easy", "topic": "interp"},
    ),
    Task(
        id="q18",
        question=(
            "Name the research group behind causal scrubbing, and the earlier, "
            "narrower technique it generalizes."
        ),
        gold_answer="Redwood Research, activation patching",
        expected_hops=2,
        metadata={"difficulty": "medium", "topic": "interp"},
    ),
]
