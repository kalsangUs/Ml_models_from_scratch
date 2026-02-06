import math

def sigmoid(x): 
    return 1 / (1 + math.exp(-x))

def simple_backpropagation(X, y, weights, learning_rate=0.1): 
    # Forward pass 
    z = sum(x * w for x, w in zip(X, weights))
    prediction = sigmoid(z)  

    # Backward pass 
    error = prediction - y 
    sigmoid_derivative = prediction * (1 - prediction) 
    # Update weights 
    new_weights = [] 
    for i, w in enumerate(weights): 
        gradient = error * sigmoid_derivative * X[i]  
        new_weights.append(w - learning_rate * gradient)  
    
    return new_weights
