import xml.etree.cElementTree as ET
import codecs
from nltk.corpus import wordnet
import networkx as nx
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Callable
import numpy as np

# Explicit types are added for better visualization of inputs and outputs of functions
class WSDInstance:
    def __init__(self, my_id, lemma, context, index):
        self.id = my_id         # id of the WSD instance
        self.lemma = lemma      # lemma of the word whose sense is to be resolved
        self.context = context  # tokenized list of words in the sentential context
        self.index = index      # index of lemma within the context

    def __str__(self):
        return '%s\t%s\t%s\t%d' % (self.id, self.lemma, ' '.join(self.context), self.index)

def to_ascii(s):
    return codecs.encode(s, 'ascii', 'ignore').decode('ascii')

def load_instances(f: str) -> Tuple[Dict[str, WSDInstance], Dict[str, WSDInstance]]:
    tree = ET.parse(f)
    root = tree.getroot()

    dev_instances = {}
    test_instances = {}

    for text in root:
        instances = dev_instances if text.attrib['id'].startswith('d001') else test_instances
        for sentence in text:
            context = [to_ascii(el.attrib['lemma']) for el in sentence]
            for i, el in enumerate(sentence):
                if el.tag == 'instance':
                    my_id = el.attrib['id']
                    lemma = to_ascii(el.attrib['lemma'])
                    instances[my_id] = WSDInstance(my_id, lemma, context, i)
    return dev_instances, test_instances

def load_key(f: str) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    dev_key = {}
    test_key = {}
    for line in open(f, encoding="utf-8"):
        if len(line) <= 1:
            continue
        doc, my_id, sense_key = line.strip().split(' ', 2)
        if doc == 'd001':
            dev_key[my_id] = sense_key.split()
        else:
            test_key[my_id] = sense_key.split()
    return dev_key, test_key


# 1: Graph-based
def graph_based_wsd(instance: WSDInstance) -> str:
    synsets = wordnet.synsets(instance.lemma)
    if not synsets:
        return None

    G = nx.Graph()
    for synset in synsets:
        G.add_node(synset)
        for related_synset in (
            synset.hypernyms()
            + synset.hyponyms()
            + synset.part_holonyms()
            + synset.part_meronyms()
            + synset.also_sees()
            + synset.similar_tos()
        ):
            G.add_edge(synset, related_synset)

    centrality = nx.pagerank(G)
    best_synset = max(synsets, key=lambda synset: centrality.get(synset, 0))
    return best_synset.lemmas()[0].key()


# 2: PMI-based
def calculate_pmi(context: List[str]) -> Dict[str, float]:
    word_freq = Counter(context)
    total_words = sum(word_freq.values())
    return {word: np.log2(freq / total_words) for word, freq in word_freq.items()}


def build_pmi_embeddings(synsets: List[wordnet.synset], context: List[str]) -> Dict[wordnet.synset, np.ndarray]:
    context_pmi = calculate_pmi(context)
    embeddings = {}
    for synset in synsets:
        embeddings[synset] = np.array([context_pmi.get(word, 0) for word in context])
    return embeddings


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    return dot_product / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-9)


def pmi_based_wsd(instance: WSDInstance) -> str:
    synsets = wordnet.synsets(instance.lemma)
    if not synsets:
        return None

    embeddings = build_pmi_embeddings(synsets, instance.context)
    context_vector = np.array([embeddings[synset] for synset in synsets]).mean(axis=0)

    best_synset = max(synsets, key=lambda synset: cosine_similarity(embeddings[synset], context_vector))
    return best_synset.lemmas()[0].key()


# 3: Hybrid
def build_synset_graph(synsets: List[wordnet.synset]) -> nx.Graph:
    G = nx.Graph()
    for synset in synsets:
        G.add_node(synset)
        for related_synset in (
            synset.hypernyms()
            + synset.hyponyms()
            + synset.part_holonyms()
            + synset.part_meronyms()
            + synset.also_sees()
            + synset.similar_tos()
        ):
            if related_synset in synsets:
                G.add_edge(synset, related_synset)
    return G


def compute_centrality(G: nx.Graph) -> Dict[wordnet.synset, float]:
    return nx.pagerank(G)


def hybrid_graph_pmi_wsd(instance: WSDInstance) -> str:
    synsets = wordnet.synsets(instance.lemma)
    if not synsets:
        return None

    G = build_synset_graph(synsets)
    centrality = compute_centrality(G)

    embeddings = build_pmi_embeddings(synsets, instance.context)
    context_vector = np.array([embeddings[synset] for synset in synsets]).mean(axis=0)

    best_synset = max(
        synsets,
        key=lambda synset: (cosine_similarity(embeddings[synset], context_vector) + centrality.get(synset, 0)),
    )

    return best_synset.lemmas()[0].key()

def evaluate_method(name: str, method: Callable, instances: Dict[str, WSDInstance], keys: Dict[str, List[str]]) -> Tuple[str, float, Callable]:
    correct = sum(
        1 for inst_id, instance in instances.items() if method(instance) in keys.get(inst_id, [])
    )
    accuracy = correct / len(instances) if instances else 0
    print(f"{name} WSD Accuracy: {accuracy:.2%}")
    return name, accuracy, method


def print_sample_classifications(method_name, method_func, instances, keys):
    """
    Prints a sample of correct and incorrect classifications, including predictions and gold sense keys.
    """
    correct_sample = None
    incorrect_sample = None

    for inst_id, instance in instances.items():
        prediction = method_func(instance)
        gold_senses = keys.get(inst_id, [])
        if prediction in gold_senses:
            if correct_sample is None:
                correct_sample = (inst_id, instance, prediction, gold_senses)
        else:
            if incorrect_sample is None:
                incorrect_sample = (inst_id, instance, prediction, gold_senses)

        if correct_sample and incorrect_sample:
            break

    print(f"\n{method_name} Classification Samples:")
    if correct_sample:
        print(f"Correctly classified: ID={correct_sample[0]}, Lemma={correct_sample[1].lemma}, "
              f"Prediction={correct_sample[2]}, Gold={correct_sample[3]}")
    else:
        print("No corect classifications found.")
    if incorrect_sample:
        print(f"Incorrectly classified: ID={incorrect_sample[0]}, Lemma={incorrect_sample[1].lemma}, "
              f"Prediction={incorrect_sample[2]}, Gold={incorrect_sample[3]}")
    else:
        print("No incorrect classification found.")


if __name__ == '__main__':
    data_f = 'multilingual-all-words.en.xml'
    key_f = 'wordnet.en.key'

    dev_instances, test_instances = load_instances(data_f)
    dev_key, test_key = load_key(key_f)

    dev_instances = {k: v for k, v in dev_instances.items() if k in dev_key}
    test_instances = {k: v for k, v in test_instances.items() if k in test_key}

    print(f"Number of develpment instances: {len(dev_instances)}")
    print(f"Number of test instances: {len(test_instances)}")

    methods = [
        ("Graph-based", graph_based_wsd),
        ("PMI-based", pmi_based_wsd),
        ("Hybrid", hybrid_graph_pmi_wsd),
    ]

    dev_results = [evaluate_method(name, method, dev_instances, dev_key) for name, method in methods]

    best_method = max(dev_results, key=lambda x: x[1])
    print(f"Best method on develpment set: {best_method[0]} with accuracy {best_method[1]:.2%}")

    print(f"Evaluating the best method: {best_method[0]} on the test set:")
    test_name, test_accuracy, test_func = evaluate_method(
        best_method[0], best_method[2], test_instances, test_key
    )

    # for method_name, method_func in methods:
    #     print_sample_classifications(method_name, method_func, test_instances, test_key)