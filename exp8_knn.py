import csv
import math
from collections import Counter

def load_dataset(filename):
    dataset = []
    with open(filename, 'r') as f:
        for row in csv.DictReader(f):
            features = [float(row['sepal_length']), float(row['sepal_width']),
                        float(row['petal_length']), float(row['petal_width'])]
            dataset.append((features, row['species']))
    return dataset

def euclidean_distance(p1, p2):
    return math.sqrt(sum((a-b)**2 for a, b in zip(p1, p2)))

def knn_classify(train_set, test_point, k):
    distances = sorted([(euclidean_distance(test_point, tp), lbl) for tp, lbl in train_set])
    neighbors = [lbl for _, lbl in distances[:k]]
    return Counter(neighbors).most_common(1)[0][0]

if __name__ == "__main__":
    data       = load_dataset('training_data_set_for_ex_8.csv')
    train_data = data[:4]
    test_data  = data[4:]
    k          = 3

    print(f"{'Actual':<15} | {'Predicted':<15} | {'Status'}")
    print("-" * 45)
    correct = 0
    for features, actual in test_data:
        predicted = knn_classify(train_data, features, k)
        status    = "CORRECT" if predicted == actual else "WRONG"
        if status == "CORRECT": correct += 1
        print(f"{actual:<15} | {predicted:<15} | {status}")

    print(f"\nAccuracy: {(correct/len(test_data))*100:.2f}%")

# OUTPUT:
# Actual          | Predicted       | Status
# ---------------------------------------------
# Iris-setosa     | Iris-setosa     | CORRECT
# Iris-versicolor | Iris-virginica  | WRONG
# Accuracy: 50.00%
