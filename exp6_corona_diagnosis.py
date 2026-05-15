import csv

def load_data(filename):
    with open(filename, 'r') as f:
        return list(csv.DictReader(f))

def diagnose(data, symptoms):
    total       = len(data)
    corona_pos  = [r for r in data if r['Corona'] == '1']
    corona_neg  = [r for r in data if r['Corona'] == '0']
    p_corona    = len(corona_pos) / total
    p_no_corona = len(corona_neg) / total

    prob_pos = p_corona
    prob_neg = p_no_corona

    for symptom, value in symptoms.items():
        pos_match = len([r for r in corona_pos if r[symptom] == str(value)])
        neg_match = len([r for r in corona_neg if r[symptom] == str(value)])
        # Laplace Smoothing (+1 numerator, +2 denominator for binary)
        prob_pos *= (pos_match + 1) / (len(corona_pos) + 2)
        prob_neg *= (neg_match + 1) / (len(corona_neg) + 2)

    return prob_pos / (prob_pos + prob_neg)

if __name__ == "__main__":
    dataset = load_data('training_data_set_for_ex_6.csv')
    patient = {'Fever': 1, 'Cough': 0, 'BreathBreath': 1}
    prob    = diagnose(dataset, patient)
    print(f"Based on symptoms {patient}:")
    print(f"Probability of Corona: {prob:.2%}")

# OUTPUT:
# Based on symptoms {'Fever': 1, 'Cough': 0, 'BreathBreath': 1}:
# Probability of Corona: 71.43%
