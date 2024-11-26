import xml.etree.cElementTree as ET
import codecs
import string
from nltk.corpus import wordnet as wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.wsd import lesk
import re

def convert_pos(pos_tag):
    noun_tags = {'NN', 'NNS', 'N', 'NPS', 'NP'}
    verb_tags = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}
    adj_tags = {'JJ', 'JJR', 'JJS'}
    adv_tags = {'RB', 'RBR', 'RBS'}
    if pos_tag in noun_tags:
        return wordnet.NOUN
    elif pos_tag in verb_tags:
        return wordnet.VERB
    elif pos_tag in adj_tags:
        return wordnet.ADJ
    elif pos_tag in adv_tags:
        return wordnet.ADV
    return None

class WSDInstance:
    def __init__(self, my_id, lemma, context, index, pos_tag):
        self.id = my_id         # id of the WSD instance
        self.lemma = lemma      # lemma of the word whose sense is to be resolved
        self.context = context  # tokenized list of words in the sentential context
        self.index = index      # index of lemma within the context
        self.pos_tag = pos_tag  # part of speech tag for the lemma

    def __str__(self):
        return '%s\t%s\t%s\t%d' % (self.id, self.lemma, ' '.join(self.context), self.index)

def load_instances(f):
    tree = ET.parse(f)
    root = tree.getroot()

    dev_instances = {}
    test_instances = {}
    instance_lemmas = set()

    for text in root:
        instances = dev_instances if text.attrib['id'].startswith('d001') else test_instances
        for sentence in text:
            context = [to_ascii(el.attrib['lemma']) for el in sentence]
            context_tokens = [clean_word(word) for word in word_tokenize(' '.join(context)) if not is_useless_punctuation(word)]
            for i, el in enumerate(sentence):
                if el.tag == 'instance':
                    my_id = el.attrib['id']
                    lemma = to_ascii(el.attrib['lemma'])
                    pos_tag = el.attrib.get('pos', 'NN')  # default to 'NN' if POS tag not providedm
                    instance_lemmas.add(lemma)
                    instances[my_id] = WSDInstance(my_id, lemma, context_tokens, i, pos_tag)
    return dev_instances, test_instances, instance_lemmas

def load_key(f):
    dev_key = {}
    test_key = {}
    with open(f) as file:
        for line in file:
            if len(line) <= 1:
                continue
            doc, my_id, sense_key = line.strip().split(' ', 2)
            if doc == 'd001':
                dev_key[my_id] = sense_key.split()
            else:
                test_key[my_id] = sense_key.split()
    return dev_key, test_key

def to_ascii(s):
    return codecs.encode(s, 'ascii', 'ignore').decode('ascii')

def clean_word(word):
    endings = ["'s", "'re", "'ve", "n't", "'ll", "'d", ".", ",", "!", "?", ":", ";", "(", ")",
               "[", "]", "{", "}", "<", ">", "\"", "'", "`", "``", "''", "-", "--", "..."]
    for ending in endings:
        if word.endswith(ending):
            return word[:-len(ending)]
    return word

def is_useless_punctuation(word):
    punctuation = set(string.punctuation)
    return all(char in punctuation for char in word)

def remove_stopwords_and_punctuation(instances):
    stop_words = set(stopwords.words('english'))
    cleaned_instances = {}

    for instance_id, instance in instances.items():
        # Tokenize context with custom handling to preserve hyphenated words
        tokenized_context = []
        for word in instance.context:
            words = re.findall(r'\b\w+(?:-\w+)*\b', word) # split only on spaces or punctuation except hyphens
            tokenized_context.extend(words)

        # Remove stopwords and punctuation
        filtered_context = [
            clean_word(word) for word in tokenized_context
            if word not in stop_words and not is_useless_punctuation(word)
        ]

        cleaned_instances[instance_id] = WSDInstance(
            instance.id, instance.lemma, filtered_context, instance.index, instance.pos_tag
        )

    return cleaned_instances

def get_synsets_for_instance_lemmas(instance_lemmas):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    lemma_synsets = {}
    for lemma in instance_lemmas:
        synsets_with_definitions = []
        for synset in wordnet.synsets(lemma):
            # Preprocess the definition by lemmatizing, lowercasing, and removing stopwords and punctuation
            definition_tokens = [
                lemmatizer.lemmatize(word.lower()) for word in word_tokenize(synset.definition())
                if word.lower() not in stop_words and not is_useless_punctuation(word)
            ]
            synsets_with_definitions.append((synset, definition_tokens))
        lemma_synsets[lemma] = synsets_with_definitions

    return lemma_synsets


def enhanced_nltk_lesk(instance, lemma_synsets):
    lemma = instance.lemma
    context_sentence = instance.context
    # Extract only synset (ignore definition tokens) from lemma_synsets
    possible_synsets = [synset for synset, _ in lemma_synsets.get(lemma, [])]
    if not possible_synsets:
        return None  # no synsets available for this lemma
    pos = convert_pos(instance.pos_tag)
    return lesk(context_sentence, lemma, pos=pos, synsets=possible_synsets)


def synset_from_sense_key(sense_key):
    try:
        return wordnet.lemma_from_key(sense_key).synset()
    except:
        return None

def most_frequent_sense(instance):
    synsets = wordnet.synsets(instance.lemma)
    return synsets[0] if synsets else None

def evaluate_accuracy(test_instances, test_key, disambiguation_function):
    correct = 0
    total = len(test_instances)

    for instance_id, instance in test_instances.items():
        predicted_synset = disambiguation_function(instance)
        correct_keys = test_key.get(instance_id, [])

        correct_synsets = {synset_from_sense_key(key) for key in correct_keys}
        if predicted_synset and predicted_synset in correct_synsets:
            correct += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy

if __name__ == '__main__':
    data_f = 'multilingual-all-words.en.xml'
    key_f = 'wordnet.en.key'

    dev_instances, test_instances, instance_lemmas = load_instances(data_f)
    dev_key, test_key = load_key(key_f)

    dev_instances = {k: v for k, v in dev_instances.items() if k in dev_key}
    test_instances = {k: v for k, v in test_instances.items() if k in test_key}
    filter_text = False

    lemma_synsets = get_synsets_for_instance_lemmas(instance_lemmas)

    mfs_accuracy = evaluate_accuracy(test_instances, test_key, most_frequent_sense)
    print(f"Most Frequent Sense Baseline Accuracy: {mfs_accuracy:.2%}")

    filter_text = True
    if filter_text:
        # Remove stopwords, punctuation, and endings from the context sentences
        dev_instances_filtered = remove_stopwords_and_punctuation(dev_instances)
        test_instances_filtered = remove_stopwords_and_punctuation(test_instances)
        lesk_accuracy_filtered = evaluate_accuracy(dev_instances_filtered, dev_key, lambda instance: enhanced_nltk_lesk(instance, lemma_synsets))
        print(f"Enhanced Lesk Algoritm Acuracy with filter_text=True: {lesk_accuracy_filtered:.2%}")

    dev_instances_unfiltered = dev_instances
    test_instances_unfiltered = test_instances
    lesk_accuracy_unfiltered = evaluate_accuracy(dev_instances_unfiltered, dev_key, lambda instance: enhanced_nltk_lesk(instance, lemma_synsets))
    print(f"Enhanced Lesk Algorithm Accuracy with filter_text=False: {lesk_accuracy_unfiltered:.2%}")

    if lesk_accuracy_filtered > lesk_accuracy_unfiltered:
        filter_text = True
        print(f"Selected filter_text=True based on dev set accuracy.")
        final_test_instances = test_instances_filtered
    else:
        filter_text = False
        print(f"Selectd filter_text=False based on dev set accuracy.")
        final_test_instances = test_instances_unfiltered

    final_lesk_accuracy = evaluate_accuracy(final_test_instances, test_key, lambda instance: enhanced_nltk_lesk(instance, lemma_synsets))
    print(f"Enhanced Lesk Algorithm Accuracy on Test Set with filter_text={filter_text}: {final_lesk_accuracy:.2%}")