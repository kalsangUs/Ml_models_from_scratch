def naive_bayes_classifier(X_train, y_train, X_test):
	# get unique classes 
	classes = list(set(y_train))
	n_samples = len(y_train)
	n_features = len(X_train[0])

	# calculate prior probabilities 
	priors = {
		c: y_train.count(c) / n_samples for c in classes 
	}

	# precompute class samples to avoid redundant computations 
	class_samples = { 
		c: [X_train[i] for i in range(len(X_train)) if y_train[i] == c] for c in classes
	}

	# calculate likelihoods P(feature | class)
	posteriors = {} 

	for c in classes: 
		# calculate P(X_test | class)
		likelihood = 1.0 
		for feature_idx, feature_val in enumerate(X_test): 
			count = sum(1 for sample in class_samples[c] if sample[feature_idx] == feature_val)
			prob = (count + 1) / (len(class_samples[c]) + 2) 
			likelihood *= prob 

		# calculate posteriror 
		posteriors[c] = priors[c] * likelihood

	# return the class with the higehst posterior probability 
	return max(posteriors, key=posteriors.get)