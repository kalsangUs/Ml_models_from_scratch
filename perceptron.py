from typing import List

def perceptron(X, y, learning_rate = 0.1, epochs=100): 
	n_features = len(X[0]) 
	weights = [0.0] * (n_features  + 1 )  # +1 for bias 

	for _ in range(epochs): 
		for i in range(len(X)): # iterate over each training example 
			x_with_bias = X[i] + [1] 
			# compute the activation 
			activation = sum(w *x for x, w in zip(weights, x_with_bias))
			prediction = 1 if activation >= 0 else -1 

			# update weights if misclassified 
			if prediction != y[1]: 
				for j in range(len(weights)): 
					weights[j] += learning_rate * y[i] * x_with_bias[j]

	return weights 

