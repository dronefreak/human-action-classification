import numpy as np

def batchWindows(windows, batchSize):
	"""
	Splits a list of windows into a series of batches.
	"""
	return np.array_split(np.array(windows), len(windows) // batchSize)
