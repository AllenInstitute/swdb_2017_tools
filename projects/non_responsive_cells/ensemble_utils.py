def make_ensemble_matrix(ensemble_pairs):
	'''Creates a matrix in which each entry indicates whether the two ensembles are correlated with each other.

	Parameters
	----------
	ensemble_pairs : a tuple (or list?) of pairs of tuples; TODO: make sure this is correct input format

	Returns
	-------
	ensemble_matrix : a matrix whose (i,j)th entry is 1 if ensembles i and j are correlated with each other and 
	0 if they are not. the dimensions of the matrix are set by the maximum ensemble number.
	'''
	# find the maximum ensemble number to set the dimensions of the ensemble matrix
	max_ensemble = max(max(pairs)[0], max(pairs, key = lambda x:x[1])[1])
	# set up the ensemble matrix
	ensemble_matrix = np.zeros((max_ensemble + 1, max_ensemble + 1), dtype='int64')
	# iterate over the pairs
	for pair in ensemble_pairs:
		# add them to both sides of the ensemble matrix
		ensemble_matrix[pair[0], pair[1]] = 1
		ensemble_matrix[pair[1], pair[0]] = 1
	return ensemble_matrix

def check_if_valid_group(ensemble_matrix, group):
	'''Checks, given an ensemble matrix, if a group of ensembles is pairwise correlated.

	Parameters
	----------
	ensemble_matrix : numpy array
		an ensemble matrix, containing only 1s and 0s, as given by make_ensemble_matrix

	Returns
	-------
	group : tuple or set
		contains the ensemble numbers of the proposed groups

	'''
	# iterate over all possible pairs from the tuple
	for pair in chain.from_iterable(combinations(set(group), r) for r in range(2,3)):
		# if that pair is not contained in the matrix, the group is out
		if ensemble_matrix[pair[0], pair[1]] == 0:
			return False
	# if we've made it this far, all pairwise relationships have satisfied the check
	return True

def groups_of_ensembles(ensemble_pairs):
	'''Extract groups of ensembles from a list of pairwise ensembles. For example, for the input
		((1,2),(2,3),(1,3),(3,4)),
	the function will return
		[{1,2}, {2,3}, {1,3}, {3,4}, {1,2,3}]
	Notice the return is a list of sets. 

	Parameters
	----------
	ensemble_pairs : a list or tuple of tuples
		contains the pairs of ensembles. function should be able to handle repeats

	Returns
	-------
	ensemble_groups : a list of sets
		a list of all ensemble groups that exist between the pairwise ensembles
	'''
	ensemble_matrix = make_ensemble_matrix(ensemble_pairs)
	ensemble_groups = []
	for cell_id in range(ensemble_matrix.shape[0]):
		connections = np.argwhere(ensemble_matrix[cell_id, :]).flatten()
		for group in chain.from_iterable(combinations(connections, r) for r in range(1, len(connections) + 1)):
			group = set(group)
			group.add(cell_id)
			if group not in ensemble_groups:
				if len(group) == 2:
					ensemble_groups.append(group)
				else:
					if check_if_valid_group(ensemble_matrix, group):
						ensemble_groups.append(group)
	return ensemble_groups