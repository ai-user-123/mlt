import csv

def load_data(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data[0], data[1:]

def candidate_elimination(examples):
    num_attributes = len(examples[0]) - 1
    S = ['0'] * num_attributes
    for ex in examples:
        if ex[-1].lower() == "yes":
            S = ex[:-1]
            break
    G = [['?' for _ in range(num_attributes)]]

    for example in examples:
        attributes, label = example[:-1], example[-1].lower()
        if label == "yes":
            for j in range(num_attributes):
                if S[j] != attributes[j]:
                    S[j] = '?'
            G = [g for g in G if all(g[k] == '?' or g[k] == attributes[k] for k in range(num_attributes))]
        else:
            new_G = []
            for g in G:
                covers = all(g[k] == '?' or g[k] == attributes[k] for k in range(num_attributes))
                if covers:
                    for k in range(num_attributes):
                        if g[k] == '?':
                            if S[k] != '?' and S[k] != attributes[k]:
                                new_h = g.copy()
                                new_h[k] = S[k]
                                if all(new_h[j] == '?' or new_h[j] == S[j] for j in range(num_attributes)):
                                    if new_h not in new_G:
                                        new_G.append(new_h)
                else:
                    new_G.append(g)
            G = new_G
    return S, G

if __name__ == "__main__":
    header, examples = load_data("training_data.csv")
    S, G = candidate_elimination(examples)
    print("--- Final Result ---")
    print("Final Specific Boundary (S):", S)
    print("Final General Boundary (G):", G)

# OUTPUT:
# --- Final Result ---
# Final Specific Boundary (S): ['sunny', 'warm', '?', 'strong', '?', '?']
# Final General Boundary (G): [['sunny', '?', '?', '?', '?', '?'], ['?', 'warm', '?', '?', '?', '?']]
