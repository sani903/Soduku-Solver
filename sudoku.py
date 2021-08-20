import cv2
import numpy as np
import inspect, sys, re, operator
from model import Trainer
from solver import Solver
import matplotlib.image as mpimg
from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
import imutils
from tensorflow.keras.preprocessing.image import img_to_array



class Detector:
	def __init__(self):
		p = re.compile("stage_(?P<idx>[0-9]+)_(?P<name>[a-zA-Z0-9_]+)")

		self.stages = list(sorted(
		map(
			lambda x: (p.fullmatch(x[0]).groupdict()['idx'], p.fullmatch(x[0]).groupdict()['name'], x[1]),
			filter(
				lambda x: inspect.ismethod(x[1]) and p.fullmatch(x[0]),
				inspect.getmembers(self))),
		key=lambda x: x[0]))

		# For storing the recognized digits
		self.digits = [ [None for i in range(9)] for j in range(9) ]

	# Takes as input 9x9 array of numpy images
	# Combines them into 1 image and returns
	# All 9x9 images need to be of same shape
	def makePreview(images):
		assert isinstance(images, list)
		assert len(images) > 0
		assert isinstance(images[0], list)
		assert len(images[0]) > 0
		assert isinstance(images[0], list)

		rows = len(images)
		cols = len(images[0])

		cellShape = images[0][0].shape

		padding = 10
		shape = (rows * cellShape[0] + (rows + 1) * padding, cols * cellShape[1] + (cols + 1) * padding)
		
		result = np.full(shape, 255, np.uint8)

		for row in range(rows):
			for col in range(cols):
				pos = (row * (padding + cellShape[0]) + padding, col * (padding + cellShape[1]) + padding)

				result[pos[0]:pos[0] + cellShape[0], pos[1]:pos[1] + cellShape[1]] = images[row][col]

		return result


	# Takes as input 9x9 array of digits
	# Prints it out on the console in the form of sudoku
	# None instead of number means that it's an empty cell
	def showSudoku(array):
		cnt = 0
		for row in array:
			if cnt % 3 == 0:
				print('+-------+-------+-------+')

			colcnt = 0
			for cell in row:
				if colcnt % 3 == 0:
					print('| ', end='')
				print('. ' if cell is None else str(cell) + ' ', end='')
				colcnt += 1
			print('|')
			cnt += 1
		print('+-------+-------+-------+')

	# Runs the detector on the image at path, and returns the 9x9 solved digits
	# if show=True, then the stage results are shown on screen
	# Corrections is an array of the kind [(1,2,9), (3,3,4) ...] which implies
	# that the digit at (1,2) is corrected to 9
	# and the digit at (3,3) is corrected to 4
	def run(self, path='assets/sudokus/sudoku1.jpg', show = False, corrections = []):
		self.path = path
		self.original = cv2.imread(path)

		# self.run_stages(show)
		result = self.solve(corrections)


		if show:
			self.showSolved()
			cv2.waitKey(0)
			cv2.destroyAllWindows()

		return result

	
	def stage_1_example1(self):
		image = cv2.cvtColor(self.original.copy(), cv2.COLOR_BGR2GRAY)
		image = cv2.GaussianBlur(image, (9,9), 0)

		self.image1 = image

		return image

	def stage_2_example2(self):
		
		self.image = cv2.resize(self.image1, (28, 28))

		self.cells = [[self.image.copy() for i in range(9)] for j in range(9)]

		return Detector.makePreview(self.cells)

	def stage_3_procimg(self):
		# apply adaptive thresholding and then invert the threshold map
		self.thresh = cv2.adaptiveThreshold(self.image1, 255,
		cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
		self.thresh = cv2.bitwise_not(self.thresh)
		# find contours in the thresholded image and sort them by size in
		# descending order
		self.cnts = cv2.findContours(self.thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
		self.cnts = imutils.grab_contours(self.cnts)
		self.cnts = sorted(self.cnts, key=cv2.contourArea, reverse=True)
		# initialize a contour that corresponds to the puzzle outline
		self.puzzleCnt = None

		# loop over the contours
		for c in self.cnts:
		# approximate the contour
			peri = cv2.arcLength(c, True)
			approx = cv2.approxPolyDP(c, 0.02 * peri, True)
			# if our approximated contour has four points, then we can
			# assume we have found the outline of the puzzle
			if len(approx) == 4:
				puzzleCnt = approx
				break
		if puzzleCnt is None:
			raise Exception(("Could not find Sudoku puzzle outline. "
			"Try debugging your thresholding and contour steps."))
		# apply a four point perspective transform to both the original
		# image and grayscale image to obtain a top-down bird's eye view
		# of the puzzle
		# puzzle = four_point_transform(self.image, puzzleCnt.reshape(4, 2))
		self.warped = four_point_transform(self.image1, puzzleCnt.reshape(4, 2))	
		



	def stage_4_extractdigit(self, cell):
		# apply automatic thresholding to the cell and then clear any
		# connected borders that touch the border of the cell
		thresh = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
		thresh = clear_border(thresh)
		# find contours in the thresholded cell
		cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
		# if no contours were found then this is an empty cell
		if len(cnts) == 0:
			return None
		# otherwise, find the largest contour in the cell and create a
		# mask for the contour
		c = max(cnts, key=cv2.contourArea)
		mask = np.zeros(thresh.shape, dtype="uint8")
		cv2.drawContours(mask, [c], -1, 255, -1)
		# compute the percentage of masked pixels relative to the total
		# area of the image
		(h, w) = thresh.shape
		percentFilled = cv2.countNonZero(mask) / float(w * h)
		# if less than 3% of the mask is filled then we are looking at
		# noise and can safely ignore the contour
		if percentFilled < 0.03:
			return None
		# apply the mask to the thresholded cell
		digit = cv2.bitwise_and(thresh, thresh, mask=mask)
		return digit



	# Solve function
	# Returns solution
	def solve(self, corrections):
		# Only upto 3 corrections allowed
		assert len(corrections) < 3
		t = Trainer()
		t.load_data()

		try:
			t.load_model()
		except:
			t.train()
		
		acc = t.test()

		assert acc > 0.9, "Accuracy not high enough"
		
		# Apply the corrections

		# Solve the sudoku
		gray = self.stage_1_example1()
		self.stage_3_procimg()
		warped = self.warped
		# a Sudoku puzzle is a 9x9 grid (81 individual cells), so we can
		# infer the location of each cell by dividing the warped image
		# into a 9x9 grid
		stepX = warped.shape[1] // 9
		stepY = warped.shape[0] // 9
		# initialize a list to store the (x, y)-coordinates of each cell
		# location
		cellLocs = []
		self.z = 0
		for y in range(0, 9):
			# initialize the current list of cell locations
			row = []
			for x in range(0, 9):
				# compute the starting and ending (x, y)-coordinates of the
				# current cell
				startX = x * stepX
				startY = y * stepY
				endX = (x + 1) * stepX
				endY = (y + 1) * stepY
				# add the (x, y)-coordinates to our cell locations list
				row.append((startX, startY, endX, endY))
				# crop the cell from the warped transform image and then
				# extract the digit from the cell
				cell = warped[startY:endY, startX:endX]
				digit = self.stage_4_extractdigit(cell)

				if digit is not None:
					# resize the cell to 28x28 pixels and then prepare the
					# cell for classification
					roi = cv2.resize(digit, (28, 28))
					roi = roi.astype("float") / 255.0
					roi = img_to_array(roi)
					roi = np.expand_dims(roi, axis=0)				
					# classify the digit and update the Sudoku board with the
					# prediction
					pred = t.model.predict(roi).argmax(axis=1)[0]
					self.digits[y][x] = pred
			cellLocs.append(row)
								
		self.answers = [[ self.digits[j][i] for i in range(9) ] for j in range(9)]
		
		s = Solver(self.answers)
		if s.solve():
			self.answers = s.digits
			return s.digits

		return [[None for i in range(9)] for j in range(9)]

	
	def showSolved(self):
		row = []
		warped = self.warped
		stepX = warped.shape[1] // 9
		stepY = warped.shape[0] // 9
		for y in range(0, 9):
			# initialize the current list of cell locations
			row = []
			for x in range(0, 9):
				# compute the starting and ending (x, y)-coordinates of the
				# current cell
				startX = x * stepX
				startY = y * stepY
				endX = (x + 1) * stepX
				endY = (y + 1) * stepY
				# add the (x, y)-coordinates to our cell locations list
				row.append((startX, startY, endX, endY))
				# crop the cell from the warped transform image and then
				# extract the digit from the cell
				cell = warped[startY:endY, startX:endX]
				digit = self.stage_4_extractdigit(cell)
				if digit is not None:
					cell = cv2.putText(cell, str(self.digits[y][x]),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
		self.z = self.z + 1
		cv2.imwrite('SolvedSoduku' + self.z, warped)


if __name__ == '__main__':
	d = Detector()
	result = d.run('assets/sudokus/sudoku1.jpg', show=True)
	print('Recognized Sudoku:')
	Detector.showSudoku(d.digits)
	print('\n\nSolved Sudoku:')
	Detector.showSudoku(result)