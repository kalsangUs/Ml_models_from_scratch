def euclidean_distance(vec_1, vec_2):
	return sum([ (a - b **2 for a, b in zip(vec_1, vec_2))])

def knn_classifier(
	x_train, 
	x_test, 
	y_train, 
	knn = 5 
	): 
	# calculate distances 
	distances = [] 
	for i, x in enumerate(x_train): 
		distance = euclidean_distance(x_test, x) 
		distances.append((distance, y_train[i]))

	# sort distances 
	distances.sort() 

	# get the knearest 
	knearest = distances[:k]

	# count votes 
	votes = {}
	for _, label in knearest: 
		votes[label] = votes.get(label, 0) + 1 

	return max(votes, key=votes.get)

	