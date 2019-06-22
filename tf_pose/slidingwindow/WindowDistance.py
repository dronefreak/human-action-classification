from .ArrayUtils import *
import numpy as np
import math

def generateDistanceMatrix(width, height):
	"""
	Generates a matrix specifying the distance of each point in a window to its centre.
	"""
	
	# Determine the coordinates of the exact centre of the window
	originX = width / 2
	originY = height / 2
	
	# Generate the distance matrix
	distances = zerosFactory((height,width), dtype=np.float)
	for index, val in np.ndenumerate(distances):
		y,x = index
		distances[(y,x)] = math.sqrt( math.pow(x - originX, 2) + math.pow(y - originY, 2) )
	
	return distances
