import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import xml.etree.ElementTree as ET
import codecs
from nltk.corpus import wordnet as wn
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split

tokenizer = AutoTokenizer.from_pretrained("kanishka/GlossBERT")
model = AutoModelForSequenceClassification.from_pretrained("kanishka/GlossBERT")
model.eval()

class WSDInstance:
    def __init__(self, my_id, lemma, context, index):
        self.id = my_id         # id of the WSD instance
        self.lemma = lemma      # lemma of the word whose sense is to be resolved
        self.context = context  # tokenized list of words in the sentential context
        self.index = index      # index of lemma within the context

    def __str__(self):
        return "%s\t%s\t%s\t%d" % (self.id, self.lemma, " ".join(self.context), self.index)


class WSDDataset(Dataset):
    def __init__(self, instances, keys, tokenizer, max_length=128):
        self.inputs = []
        self.labels = []
        for instance_id, instance in instances.items():
            context_sentence = " ".join(instance.context)
            correct_keys = keys.get(instance_id, [])
            correct_synsets = {wn.synset_from_sense_key(key) for key in correct_keys if wn.synset_from_sense_key(key)}
            candidate_synsets = wn.synsets(instance.lemma)
            for synset in candidate_synsets:
                gloss = synset.definition()
                # Input for the model as a pair (context, gloss)
                encoding = tokenizer(
                    text=context_sentence,
                    text_pair=gloss,
                    truncation=True,
                    max_length=max_length,
                    padding="max_length",
                    return_tensors="pt",
                )
                label = 1 if synset in correct_synsets else 0
                self.inputs.append({key: val.squeeze() for key, val in encoding.items()})
                self.labels.append(label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = self.inputs[idx]
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def train(
    model,
    train_dataset,
    val_dataset=None,
    epochs=3,
    batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    max_grad_norm=1.0,
    patience=2,
):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * len(train_loader) * epochs), num_training_steps=len(train_loader) * epochs
    )

    best_val_accuracy = 0
    no_improvement_epochs = 0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()
            batch = {key: val.to(device) for key, val in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()

        print(f"Average training loss: {total_loss / len(train_loader):.4f}")

        if val_loader:
            val_loss, val_accuracy = validate(model, val_loader, device)
            print(f"Validation loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2%}")
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                no_improvement_epochs = 0
            else:
                no_improvement_epochs += 1
                if no_improvement_epochs >= patience:
                    print(f"Early stopping triggerd after {patience} epochs.")
                    break


def validate(model, val_loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            batch = {key: val.to(device) for key, val in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            predictions = torch.argmax(outputs.logits, dim=1)
            correct += (predictions == batch["labels"]).sum().item()
            total += batch["labels"].size(0)

    return total_loss / len(val_loader), correct / total


def evaluate_with_examples(disambiguation_func, instances, keys, max_examples=3):
    correct, total = 0, 0
    correct_examples = []
    incorrect_examples = []

    for instance_id, instance in tqdm(instances.items(), desc="Evaluating Instances", unit="instance"):
        predicted_key = disambiguation_func(instance)
        correct_keys = keys.get(instance_id, [])

        if predicted_key:
            is_correct = predicted_key in correct_keys
            if is_correct:
                correct += 1
                if len(correct_examples) < max_examples:
                    correct_examples.append((instance_id, instance.lemma, predicted_key, correct_keys))
            else:
                if len(incorrect_examples) < max_examples:
                    incorrect_examples.append((instance_id, instance.lemma, predicted_key, correct_keys))
        total += 1

    print(f"\nCorrectly disambiguated: {correct}/{total} ({100 * correct / total:.2f}%)")
    print("\nExamples of Correct Predictions:")
    for example in correct_examples:
        print(f"Instance: {example[0]}, Lemma: {example[1]}, Predicted: {example[2]}, Gold: {example[3]}")

    print("\nExamples of Incorect Predictions:")
    for example in incorrect_examples:
        print(f"Instance: {example[0]}, Lemma: {example[1]}, Predicted: {example[2]}, Gold: {example[3]}")

    return correct / total


def disambiguate(instance):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    context_sentence = " ".join(instance.context)
    candidate_synsets = wn.synsets(instance.lemma)

    if not candidate_synsets:
        return None

    inputs = [
        tokenizer(context_sentence, synset.definition(), return_tensors="pt", truncation=True, max_length=128).to(device)
        for synset in candidate_synsets
    ]

    with torch.no_grad():
        logits = [model(**inp).logits[0, 1].item() for inp in inputs]

    best_synset_idx = np.argmax(logits)
    return candidate_synsets[best_synset_idx].lemmas()[0].key()


def load_instances(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    dev_instances, test_instances = {}, {}

    for text in root:
        instances = dev_instances if text.attrib["id"].startswith("d001") else test_instances
        for sentence in text:
            context = [to_ascii(el.attrib["lemma"]) for el in sentence]
            for i, el in enumerate(sentence):
                if el.tag == "instance":
                    instances[el.attrib["id"]] = WSDInstance(el.attrib["id"], to_ascii(el.attrib["lemma"]), context, i)

    return dev_instances, test_instances


def load_key(file_path):
    dev_key, test_key = {}, {}
    with open(file_path, "r") as file:
        for line in file:
            doc, my_id, *sense_keys = line.strip().split()
            target = dev_key if doc == "d001" else test_key
            target[my_id] = sense_keys
    return dev_key, test_key


def to_ascii(s):
    return codecs.encode(s, "ascii", "ignore").decode("ascii")


if __name__ == "__main__":
    data_f = "multilingual-all-words.en.xml"
    key_f = "wordnet.en.key"

    test_instances, dev_instances = load_instances(data_f)
    test_key, dev_key = load_key(key_f)

    dev_instances = {k: v for k, v in dev_instances.items() if k in dev_key}
    test_instances = {k: v for k, v in test_instances.items() if k in test_key}

    train_items, val_items = train_test_split(list(dev_instances.items()), test_size=0.1, random_state=42)
    train_instances, val_instances = dict(train_items), dict(val_items)

    train_dataset = WSDDataset(train_instances, dev_key, tokenizer)
    val_dataset = WSDDataset(val_instances, dev_key, tokenizer)

    train(
        model,
        train_dataset,
        val_dataset=val_dataset,
        epochs=4,
        batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        patience=3,
    )

    print("\nEvaluating on Devlopment Set:") # after fine-tuning evaluate one last time on dev set
    evaluate_with_examples(disambiguate, dev_instances, dev_key)

    print("\nEvaluating on Test Set:") # generalization to unseen data
    evaluate_with_examples(disambiguate, test_instances, test_key)