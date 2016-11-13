import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_regression, mutual_info_regression


def extract_data(data_file, feature_title=True):
	with open(data_file, 'r') as handle:
	    parsed = json.load(handle)

	y = list(map(lambda x:x[5], parsed))[1:]
	Y = np.array(list(map(float,y)))

	x1 = list(map(lambda x:(x[1:3]), parsed))[1:]
	x1 = [list(map(float,x)) for x in x1]
	x2 = list(map(lambda x:(x[6:]), parsed))[1:]
	x2 = [list(map(float,x)) for x in x2]
	X = np.array([a + b for a, b in zip(x1,x2)])

	feature_titles = parsed[0][1:3] + parsed[0][6:]

	if feature_title:
		return X, Y, feature_titles
	else:
		return X, Y

def var_corr(features, plot=False):
	cm = np.corrcoef(features.T)
	if plot:
		plt.imshow(cm,interpolation='nearest')
		plt.colorbar()
		plt.show()
	else:
		return cm
		
def feature_ranks(X, Y):
	f_test, _ = f_regression(X, Y)
	f_test /= np.max(f_test)
	mi = mutual_info_regression(X, Y)
	mi /= np.max(mi)

	w = 1
	combined_features = f_test + w * mi
	f_ranks = np.argsort(combined_features)

	return np.flipud(f_ranks)

def sel_KBest(features, f_ranks, K):
	X = features.T
	X = np.delete(X, f_ranks[K:], axis=0)
	return X.T

def import_data(data_file, K):
	X, Y, f_t = extract_data(data_file)
	f_ranks = feature_ranks(X, Y)
	return sel_KBest(X, f_ranks, K), Y, f_t

