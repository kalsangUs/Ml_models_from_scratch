import math 


def sigmoid(x) -> float: 
	return 1 / ( 1 + math.exp(-x))



def neural_network_forward_pass(x, w_1, w_2) -> float:
	# 2 layer forward pass
    hidden = []
    
    # Hidden layer calculations
    for neuron_weights in w_1:
        z = sum(x_i * w_i for x_i, w_i in zip(x, neuron_weights))
        activation = sigmoid(z)  # Assuming sigmoid is defined elsewhere
        hidden.append(activation)
    
    # Output layer calculations
    output = sum(h * w for h, w in zip(hidden, w_2))
    final_output = sigmoid(output)
    
    return final_output

if __name__ == "__main__": 
	x = [0.5, -0.2, 0.8]
	w_1 = [
		[0.3, 0.6, -0.1],  # weights for the first hidden neuron 
		[-0.4, 0.2, 0.5],  # weights for the second hidden neuron 
		[0.1, -0.3, 0.2] # weights for the third hidden neuron 
	]

	w_2 = [0.7, -0.5, 0.5] # weights for the output layer(1 output )

	output = neural_network_forward_pass(x, w_1, w_2)
	print(output)