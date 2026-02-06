def linear_regression_with_gradient_descent(X, y, learning_rate = 0.1, iterations = 1000): 
	n = len(X)
	w = 0.0
	b = 0.0 

	for _ in range(iterations): 
		# compute predcitions 
		y_pred = [w * x + b for x in X]

		# compute gradients 
		dw = (-2/n) * sum([(y[i] - y_pred[i]) * X[i] for i in range(n)])
		db = (-2/n) * sum([(y[i] - y_pred[i]) for i in range(n)])

		# update weights 
		w -= learning_rate * dw 
		b -= learning_rate * db 

	return w, b 

