import csv

def load_csv(filename):
    with open(filename, 'r') as f:
        data = list(csv.reader(f))
    return data[1:], data[0]

def split_dataset(dataset, ratio):
    n = int(len(dataset) * ratio)
    return dataset[:n], dataset[n:]

def separate_by_class(dataset):
    separated = {}
    for row in dataset:
        separated.setdefault(row[-1], []).append(row)
    return separated

def calculate_attribute_probabilities(dataset):
    total = len(dataset)
    probs = {}
    for i in range(len(dataset[0]) - 1):
        probs[i] = {}
        for row in dataset:
            probs[i][row[i]] = probs[i].get(row[i], 0) + 1
        for val in probs[i]:
            probs[i][val] /= total
    return probs, total

def predict(summaries, class_probs, input_vector):
    probabilities = {}
    for cls, data in summaries.items():
        probabilities[cls] = class_probs[cls]
        for i, val in enumerate(input_vector):
            probabilities[cls] *= data['attr_probs'][i].get(val, 0.0001)
    return max(probabilities, key=probabilities.get)

def get_accuracy(test_set, summaries, class_probs):
    correct = sum(1 for row in test_set if predict(summaries, class_probs, row[:-1]) == row[-1])
    return (correct / len(test_set)) * 100

if __name__ == "__main__":
    dataset, header = load_csv('training_data_set_for_ex_4.csv')
    train_set, test_set = split_dataset(dataset, 0.8)

    separated = separate_by_class(train_set)
    summaries, class_probs = {}, {}
    for cls, instances in separated.items():
        attr_probs, count = calculate_attribute_probabilities(instances)
        summaries[cls] = {'attr_probs': attr_probs}
        class_probs[cls] = count / len(train_set)

    print(f"Dataset: {len(dataset)} rows | Train: {len(train_set)} | Test: {len(test_set)}")
    print(f"Accuracy: {get_accuracy(test_set, summaries, class_probs)}%")

    new_sample = ['High', 'Low', 'Yes']
    print(f"New Sample {new_sample} predicted as: {predict(summaries, class_probs, new_sample)}")

# OUTPUT:
# Dataset: 10 rows | Train: 8 | Test: 2
# Accuracy: 50.0%
# New Sample ['High', 'Low', 'Yes'] predicted as: Yes
