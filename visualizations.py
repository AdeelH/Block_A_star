from time import perf_counter
from collections import defaultdict
import matplotlib.pyplot as plt

import numpy as np
from Block import *
from BlockMap import *
from PriorityQueue import *
from LDDB import *


def visualize_map(Map, start=None, goal=None, path=None):

	X = Map + 0.
	X[X == 1] = 5
	X[X == 0] = 1
	if path:
		for y, x in path:
			X[y, x] = 2
	if start:
		X[start[0], start[1]] = 0
	if goal:
		X[goal[0], goal[1]] = 4

	fig, ax = plt.subplots()
	ax.imshow(X, interpolation='nearest', cmap='Set3')

	numrows, numcols = X.shape


	def format_coord(x, y):
	    col = int(x + 0.5)
	    row = int(y + 0.5)
	    if col >= 0 and col < numcols and row >= 0 and row < numrows:
	        z = X[row, col]
	        return 'x=%1.4f, y=%1.4f, z=%1.4f' % (x, y, z)
	    else:
	        return 'x=%1.4f, y=%1.4f' % (x, y)

	ax.format_coord = format_coord
	plt.show()


def pretty_print(Map, block_size=None, path=[], start=None, goal=None):
	if path:
		path = set(path)
	else:
		path = [None]

	show_blocks = block_size is not None

	for i, row in enumerate(Map):
		if show_blocks and i % block_size == 0:
			print()
		for j, _ in enumerate(row):
			if show_blocks and j % block_size == 0:
				print(' ', end='')
			n_ = (i, j)
			if n_ == start:
				print(' S', end=' ')
			elif n_ == goal:
				print(' G', end=' ')
			elif n_ in path:
				print(' x', end=' ')
			elif Map[i][j] == 0:
				print('\u2591\u2591', end=' ')
			else:
				print('\u2588\u2588', end=' ')
		print('\n')

