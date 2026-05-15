import csv
import math
import random

def load_nn_data(filename):
    inputs, targets = [], []
    with open(filename, 'r') as f:
        for row in csv.DictReader(f):
            inputs.append([float(row['input1']), float(row['input2'])])
            targets.append([float(row['target'])])
    return inputs, targets

def sigmoid(x): return 1 / (1 + math.exp(-x))
def sigmoid_derivative(x): return x * (1 - x)

class NeuralNetwork:
    def __init__(self, inp, hid, out):
        self.w_ih = [[random.uniform(-1,1) for _ in range(hid)] for _ in range(inp)]
        self.w_ho = [[random.uniform(-1,1) for _ in range(out)] for _ in range(hid)]
        self.bias_h = [random.uniform(-1,1) for _ in range(hid)]
        self.bias_o = [random.uniform(-1,1) for _ in range(out)]

    def train(self, inputs, targets, lr):
        # Forward Pass
        h_in  = [sum(inputs[i]*self.w_ih[i][j] for i in range(len(inputs))) + self.bias_h[j] for j in range(len(self.bias_h))]
        h_out = [sigmoid(x) for x in h_in]
        o_in  = [sum(h_out[j]*self.w_ho[j][k] for j in range(len(h_out))) + self.bias_o[k] for k in range(len(self.bias_o))]
        final = [sigmoid(x) for x in o_in]

        # Backpropagation
        out_err  = [targets[k] - final[k] for k in range(len(targets))]
        out_grad = [out_err[k] * sigmoid_derivative(final[k]) for k in range(len(final))]
        h_err    = [sum(out_grad[k] * self.w_ho[j][k] for k in range(len(out_grad))) for j in range(len(h_out))]
        h_grad   = [h_err[j] * sigmoid_derivative(h_out[j]) for j in range(len(h_out))]

        # Update Weights
        for j in range(len(h_out)):
            for k in range(len(final)):
                self.w_ho[j][k] += out_grad[k] * h_out[j] * lr
        for i in range(len(inputs)):
            for j in range(len(h_out)):
                self.w_ih[i][j] += h_grad[j] * inputs[i] * lr
        return final

if __name__ == "__main__":
    inputs, targets = load_nn_data("training_data_set_for_ex_3.csv")
    nn = NeuralNetwork(2, 2, 1)
    lr, epochs = 0.1, 20000

    print("Training Neural Network via Backpropagation...")
    for epoch in range(epochs):
        total_error = 0
        for i in range(len(inputs)):
            pred = nn.train(inputs[i], targets[i], lr)
            total_error += (targets[i][0] - pred[0]) ** 2
        if epoch % 2000 == 0:
            print(f"Epoch {epoch:5d} | MSE: {total_error/len(inputs):.6f}")

    print("\nFinal Results:")
    for i in range(len(inputs)):
        pred = nn.train(inputs[i], targets[i], 0)
        print(f"Input: {inputs[i]} -> Predicted: {pred[0]:.4f} (Target: {targets[i][0]})")

# OUTPUT:
# Epoch     0 | MSE: 0.283638
# Epoch  2000 | MSE: 0.235122
# ...
# Epoch 18000 | MSE: 0.133833
# Input: [0.0, 0.0] -> Predicted: 0.1466 (Target: 0.0)
# Input: [0.0, 1.0] -> Predicted: 0.9823 (Target: 1.0)
