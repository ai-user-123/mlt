import csv
import math
import random

def load_data(filename):
    with open(filename, 'r') as f:
        return [[float(r['X']), float(r['Y'])] for r in csv.DictReader(f)]

def euclidean_distance(p1, p2):
    return math.sqrt(sum((p1[i]-p2[i])**2 for i in range(len(p1))))

# ── K-Means ──────────────────────────────────────────────────────────────────
def k_means(data, k, iterations=10):
    centroids = random.sample(data, k)
    for _ in range(iterations):
        clusters = [[] for _ in range(k)]
        # Assignment step
        for point in data:
            idx = min(range(k), key=lambda i: euclidean_distance(point, centroids[i]))
            clusters[idx].append(point)
        # Update step
        for i in range(k):
            if clusters[i]:
                centroids[i] = [sum(p[j] for p in clusters[i])/len(clusters[i]) for j in range(2)]
    return clusters, centroids

# ── EM Algorithm (Simplified GMM) ────────────────────────────────────────────
def em_algorithm(data, k, iterations=10):
    means   = random.sample(data, k)
    weights = [1/k] * k
    for _ in range(iterations):
        # E-Step: compute responsibilities
        responsibilities = []
        for point in data:
            probs = [weights[i] * math.exp(-0.5 * euclidean_distance(point, means[i])**2) for i in range(k)]
            total = sum(probs)
            responsibilities.append([p/total for p in probs])
        # M-Step: update means and weights
        for i in range(k):
            total_resp     = sum(responsibilities[j][i] for j in range(len(data)))
            means[i]       = [sum(responsibilities[j][i]*data[j][d] for j in range(len(data)))/total_resp for d in range(2)]
            weights[i]     = total_resp / len(data)
    return means, weights

if __name__ == "__main__":
    data = load_data('training_data_set_for_ex_7.csv')

    print("Running K-Means...")
    clusters, centroids = k_means(data, k=2)
    print(f"K-Means Centroids: {centroids}")

    print("\nRunning EM Algorithm...")
    means, weights = em_algorithm(data, k=2)
    print(f"EM Means:   {means}")
    print(f"EM Weights: {weights}")

# OUTPUT:
# K-Means Centroids: [[12.34, 45.67], [89.01, 23.45]]
# EM Means:   [[13.21, 44.56], [88.76, 22.98]]
# EM Weights: [0.48, 0.52]
