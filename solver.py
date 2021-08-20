from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
from model import Trainer


# None element in the 9x9 array represents empty cell
class Solver:
	def __init__(self, digits):
		self.digits = digits
		
	
	# Returns True if sudoku admits a solution
	# False otherwise
	# Solved sudoku can be found in self.digits
	def findNextCellToFill(self, i, j):
		for x in range(i, 9):
			for y in range(j, 9):
				if self.digits[x][y] is None:
					return x, y
		for x in range(0, 9):
			for y in range(0, 9):
				if self.digits[x][y] is None:
					return x, y
		return -1, -1					


	def isValid(self, i, j, e):
		rowOk = all([e != self.digits[i][x] for x in range(9)])
		if rowOk:
			columnOk = all([e != self.digits[x][j] for x in range(9)])
			if columnOk:
				# finding the top left x,y co-ordinates of the section containing the i,j cell
				secTopX, secTopY = 3 *(i//3), 3 *(j//3) #floored quotient should be used here. 
				for x in range(secTopX, secTopX+3):
					for y in range(secTopY, secTopY+3):
						if self.digits[x][y] == e:
							return False
				return True
		return False	
		

	def solve(self):
		i = 0
		j = 0
		i,j = self.findNextCellToFill(i, j)		
		if i == -1:
			return True
		for e in range(1,10):
			if self.isValid(i,j,e):
				self.digits[i][j] = e
				if self.solve():
					return True
				# Undo the current cell for backtracking
				self.digits[i][j] = 0
				return False
				

	


