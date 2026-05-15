import csv
import math
import re
from collections import Counter

def tokenize(text):
    return re.findall(r'\w+', text.lower())

def load_data(filename):
    with open(filename, 'r') as f:
        return [(tokenize(row['text']), row['label']) for row in csv.DictReader(f)]

def train_naive_bayes(train_data):
    class_counts = Counter()
    word_counts  = {}
    vocab        = set()
    for words, label in train_data:
        class_counts[label] += 1
        word_counts.setdefault(label, Counter())
        for word in words:
            word_counts[label][word] += 1
            vocab.add(word)
    return class_counts, word_counts, vocab

def classify(doc, class_counts, word_counts, vocab):
    total = sum(class_counts.values())
    best_label, max_prob = None, -float('inf')
    for label in class_counts:
        log_prob = math.log(class_counts[label] / total)
        total_words = sum(word_counts[label].values())
        for word in doc:
            if word in vocab:
                count = word_counts[label].get(word, 0) + 1
                log_prob += math.log(count / (total_words + len(vocab)))
        if log_prob > max_prob:
            max_prob, best_label = log_prob, label
    return best_label

def calculate_metrics(test_data, class_counts, word_counts, vocab):
    target = 'tech'
    tp = fp = fn = tn = 0
    for words, actual in test_data:
        predicted = classify(words, class_counts, word_counts, vocab)
        if predicted == target and actual == target:   tp += 1
        elif predicted == target and actual != target: fp += 1
        elif predicted != target and actual == target: fn += 1
        else:                                          tn += 1
    accuracy  = (tp + tn) / len(test_data)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    return accuracy, precision, recall

if __name__ == "__main__":
    data  = load_data('training_data_set_for_ex_5.csv')
    split = int(len(data) * 0.75)
    train_set, test_set = data[:split], data[split:]

    class_counts, word_counts, vocab = train_naive_bayes(train_set)
    acc, prec, rec = calculate_metrics(test_set, class_counts, word_counts, vocab)

    print("Metrics for class 'tech':")
    print(f"Accuracy:  {acc:.2f}")
    print(f"Precision: {prec:.2f}")
    print(f"Recall:    {rec:.2f}")

# OUTPUT:
# Metrics for class 'tech':
# Accuracy:  0.50
# Precision: 0.50
# Recall:    1.00
