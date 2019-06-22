import math, mmap, tempfile
import numpy as np
import psutil

def _requiredSize(shape, dtype):
	"""
	Determines the number of bytes required to store a NumPy array with
	the specified shape and datatype.
	"""
	return math.floor(np.prod(np.asarray(shape, dtype=np.uint64)) * np.dtype(dtype).itemsize)


class TempfileBackedArray(np.ndarray):
	"""
	A NumPy ndarray that uses a memory-mapped temp file as its backing 
	"""
	
	def __new__(subtype, shape, dtype=float, buffer=None, offset=0, strides=None, order=None, info=None):
		
		# Determine the size in bytes required to hold the array
		numBytes = _requiredSize(shape, dtype)
		
		# Create the temporary file, resize it, and map it into memory
		tempFile = tempfile.TemporaryFile()
		tempFile.truncate(numBytes)
		buf = mmap.mmap(tempFile.fileno(), numBytes, access=mmap.ACCESS_WRITE)
		
		# Create the ndarray with the memory map as the underlying buffer
		obj = super(TempfileBackedArray, subtype).__new__(subtype, shape, dtype, buf, 0, None, order)
		
		# Attach the file reference to the ndarray object
		obj._file = tempFile
		return obj
	
	def __array_finalize__(self, obj):
		if obj is None: return
		self._file = getattr(obj, '_file', None)


def arrayFactory(shape, dtype=float):
	"""
	Creates a new ndarray of the specified shape and datatype, storing
	it in memory if there is sufficient available space or else using
	a memory-mapped temporary file to provide the underlying buffer.
	"""
	
	# Determine the number of bytes required to store the array
	requiredBytes = _requiredSize(shape, dtype)
	
	# Determine if there is sufficient available memory
	vmem = psutil.virtual_memory()
	if vmem.available > requiredBytes:
		return np.ndarray(shape=shape, dtype=dtype)
	else:
		return TempfileBackedArray(shape=shape, dtype=dtype)


def zerosFactory(shape, dtype=float):
	"""
	Creates a new NumPy array using `arrayFactory()` and fills it with zeros.
	"""
	arr = arrayFactory(shape=shape, dtype=dtype)
	arr.fill(0)
	return arr


def arrayCast(source, dtype):
	"""
	Casts a NumPy array to the specified datatype, storing the copy
	in memory if there is sufficient available space or else using a
	memory-mapped temporary file to provide the underlying buffer.
	"""
	
	# Determine the number of bytes required to store the array
	requiredBytes = _requiredSize(source.shape, dtype)
	
	# Determine if there is sufficient available memory
	vmem = psutil.virtual_memory()
	if vmem.available > requiredBytes:
		return source.astype(dtype, subok=False)
	else:
		dest = arrayFactory(source.shape, dtype)
		np.copyto(dest, source, casting='unsafe')
		return dest


def determineMaxWindowSize(dtype, limit=None):
	"""
	Determines the largest square window size that can be used, based on
	the specified datatype and amount of currently available system memory.
	
	If `limit` is specified, then this value will be returned in the event
	that it is smaller than the maximum computed size.
	"""
	vmem = psutil.virtual_memory()
	maxSize = math.floor(math.sqrt(vmem.available / np.dtype(dtype).itemsize))
	if limit is None or limit >= maxSize:
		return maxSize
	else:
		return limit
