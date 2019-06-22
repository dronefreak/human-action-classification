from .SlidingWindow import generate
from .Batching import batchWindows
import numpy as np

def mergeWindows(data, dimOrder, maxWindowSize, overlapPercent, batchSize, transform, progressCallback = None):
	"""
	Generates sliding windows for the specified dataset and applies the specified
	transformation function to each window. Where multiple overlapping windows
	include an element of the input dataset, the overlap is resolved by computing
	the mean transform result value for that element.
	
	Irrespective of the order of the dimensions of the input dataset, the
	transformation function should return a NumPy array with dimensions
	[batch, height, width, resultChannels].
	
	If a progress callback is supplied, it will be called immediately before
	applying the transformation function to each batch of windows. The callback
	should accept the current batch index and number of batches as arguments.
	"""
	
	# Determine the dimensions of the input data
	sourceWidth = data.shape[dimOrder.index('w')]
	sourceHeight = data.shape[dimOrder.index('h')]
	
	# Generate the sliding windows and group them into batches
	windows = generate(data, dimOrder, maxWindowSize, overlapPercent)
	batches = batchWindows(windows, batchSize)
	
	# Apply the transform to the first batch of windows and determine the result dimensionality
	exemplarResult = transform(data, batches[0])
	resultDimensions = exemplarResult.shape[ len(exemplarResult.shape) - 1 ]
	
	# Create the matrices to hold the sums and counts for the transform result values
	sums = np.zeros((sourceHeight, sourceWidth, resultDimensions), dtype=np.float)
	counts = np.zeros((sourceHeight, sourceWidth), dtype=np.uint32)
	
	# Iterate over the batches and apply the transformation function to each batch
	for batchNum, batch in enumerate(batches):
		
		# If a progress callback was supplied, call it
		if progressCallback != None:
			progressCallback(batchNum, len(batches))
		
		# Apply the transformation function to the current batch
		batchResult = transform(data, batch)
		
		# Iterate over the windows in the batch and update the sums matrix
		for windowNum, window in enumerate(batch):
			
			# Create views into the larger matrices that correspond to the current window
			windowIndices = window.indices(False)
			sumsView = sums[windowIndices]
			countsView = counts[windowIndices]
			
			# Update the result sums for each of the dataset elements in the window
			sumsView[:] += batchResult[windowNum]
			countsView[:] += 1
	
	# Use the sums and the counts to compute the mean values
	for dim in range(0, resultDimensions):
		sums[:,:,dim] /= counts
	
	# Return the mean values
	return sums
