import numpy as np
import math

def cropRect(rect, cropTop, cropBottom, cropLeft, cropRight):
	"""
	Crops a rectangle by the specified number of pixels on each side.
	
	The input rectangle and return value are both a tuple of (x,y,w,h).
	"""
	
	# Unpack the rectangle
	x, y, w, h = rect
	
	# Crop by the specified value
	x += cropLeft
	y += cropTop
	w -= (cropLeft + cropRight)
	h -= (cropTop + cropBottom)
	
	# Re-pack the padded rect
	return (x,y,w,h)


def padRect(rect, padTop, padBottom, padLeft, padRight, bounds, clipExcess = True):
	"""
	Pads a rectangle by the specified values on each individual side,
	ensuring the padded rectangle falls within the specified bounds.
	
	The input rectangle, bounds, and return value are all a tuple of (x,y,w,h).
	"""
	
	# Unpack the rectangle
	x, y, w, h = rect
	
	# Pad by the specified value
	x -= padLeft
	y -= padTop
	w += (padLeft + padRight)
	h += (padTop + padBottom)
	
	# Determine if we are clipping overflows/underflows or
	# shifting the centre of the rectangle to compensate
	if clipExcess == True:
		
		# Clip any underflows
		x = max(0, x)
		y = max(0, y)
		
		# Clip any overflows
		overflowY = max(0, (y + h) - bounds[0])
		overflowX = max(0, (x + w) - bounds[1])
		h -= overflowY
		w -= overflowX
		
	else:
		
		# Compensate for any underflows
		underflowX = max(0, 0 - x)
		underflowY = max(0, 0 - y)
		x += underflowX
		y += underflowY
		
		# Compensate for any overflows
		overflowY = max(0, (y + h) - bounds[0])
		overflowX = max(0, (x + w) - bounds[1])
		x -= overflowX
		w += overflowX
		y -= overflowY
		h += overflowY
		
		# If there are still overflows or underflows after our
		# modifications, we have no choice but to clip them
		x, y, w, h = padRect((x,y,w,h), 0, 0, 0, 0, bounds, True)
	
	# Re-pack the padded rect
	return (x,y,w,h)


def cropRectEqually(rect, cropping):
	"""
	Crops a rectangle by the specified number of pixels on all sides.
	
	The input rectangle and return value are both a tuple of (x,y,w,h).
	"""
	return cropRect(rect, cropping, cropping, cropping, cropping)


def padRectEqually(rect, padding, bounds, clipExcess = True):
	"""
	Applies equal padding to all sides of a rectangle,
	ensuring the padded rectangle falls within the specified bounds.
	
	The input rectangle, bounds, and return value are all a tuple of (x,y,w,h).
	"""
	return padRect(rect, padding, padding, padding, padding, bounds, clipExcess)


def squareAspect(rect):
	"""
	Crops either the width or height, as necessary, to make a rectangle into a square.
	
	The input rectangle and return value are both a tuple of (x,y,w,h).
	"""
	
	# Determine which dimension needs to be cropped
	x,y,w,h = rect
	if w > h:
		cropX = (w - h) // 2
		return cropRect(rect, 0, 0, cropX, cropX)
	elif w < h:
		cropY = (h - w) // 2
		return cropRect(rect, cropY, cropY, 0, 0)
	
	# Already a square
	return rect


def fitToSize(rect, targetWidth, targetHeight, bounds):
	"""
	Pads or crops a rectangle as necessary to achieve the target dimensions,
	ensuring the modified rectangle falls within the specified bounds.
	
	The input rectangle, bounds, and return value are all a tuple of (x,y,w,h).
	"""
	
	# Determine the difference between the current size and target size
	x,y,w,h = rect
	diffX = w - targetWidth
	diffY = h - targetHeight
	
	# Determine if we are cropping or padding the width
	if diffX > 0:
		cropLeft  = math.floor(diffX / 2)
		cropRight = diffX - cropLeft
		x,y,w,h   = cropRect((x,y,w,h), 0, 0, cropLeft, cropRight)
	elif diffX < 0:
		padLeft  = math.floor(abs(diffX) / 2)
		padRight = abs(diffX) - padLeft
		x,y,w,h  = padRect((x,y,w,h), 0, 0, padLeft, padRight, bounds, False)
	
	# Determine if we are cropping or padding the height
	if diffY > 0:
		cropTop    = math.floor(diffY / 2)
		cropBottom = diffY - cropTop
		x,y,w,h    = cropRect((x,y,w,h), cropTop, cropBottom, 0, 0)
	elif diffY < 0:
		padTop    = math.floor(abs(diffY) / 2)
		padBottom = abs(diffY) - padTop
		x,y,w,h   = padRect((x,y,w,h), padTop, padBottom, 0, 0, bounds, False)
	
	return (x,y,w,h)
