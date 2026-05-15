import csv
import math
import pprint

def load_data(filename):
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        return [row for row in reader]

def calculate_entropy(data):
    if not data: return 0
    total = len(data)
    counts = {}
    for row in data:
        counts[row['PlayTennis']] = counts.get(row['PlayTennis'], 0) + 1
    return sum(-( c/total) * math.log2(c/total) for c in counts.values())

def calculate_information_gain(data, attribute):
    total = len(data)
    subsets = {}
    for row in data:
        subsets.setdefault(row[attribute], []).append(row)
    weighted_entropy = sum((len(s)/total) * calculate_entropy(s) for s in subsets.values())
    return calculate_entropy(data) - weighted_entropy

def id3(data, attributes):
    labels = [row['PlayTennis'] for row in data]
    if len(set(labels)) == 1: return labels[0]
    if not attributes: return max(set(labels), key=labels.count)

    best_attr = max(attributes, key=lambda a: calculate_information_gain(data, a))
    tree = {best_attr: {}}
    remaining = [a for a in attributes if a != best_attr]
    for val in set(row[best_attr] for row in data):
        subset = [row for row in data if row[best_attr] == val]
        tree[best_attr][val] = id3(subset, remaining)
    return tree

def classify(sample, tree):
    if not isinstance(tree, dict): return tree
    root = list(tree.keys())[0]
    val = sample.get(root)
    return classify(sample, tree[root][val]) if val in tree[root] else "Unknown"

if __name__ == "__main__":
    data = load_data("training_data_set_for_ex_2.csv")
    attributes = list(data[0].keys())
    attributes.remove('PlayTennis')

    print("Building Decision Tree...")
    tree = id3(data, attributes)
    print("\nGenerated Decision Tree:")
    pprint.pprint(tree)

    new_sample = {'Outlook': 'Rain', 'Temperature': 'Cool', 'Humidity': 'High', 'Wind': 'Strong'}
    print(f"\nClassifying: {new_sample}")
    print(f"Prediction: {classify(new_sample, tree)}")

# OUTPUT:
# Generated Decision Tree:
# {'Outlook': {'Overcast': 'Yes',
#              'Rain': {'Wind': {'Strong': 'No', 'Weak': 'Yes'}},
#              'Sunny': {'Humidity': {'High': 'No', 'Normal': 'Yes'}}}}
# Prediction: No
