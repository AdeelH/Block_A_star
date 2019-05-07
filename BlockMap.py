from time import perf_counter
from collections import defaultdict

import numpy as np
from Block import Block, list_to_block_idx
from PriorityQueue import PriorityQueue


class BlockMap(object):
	def __init__(self, Map, block_size):
		"""Holds the grid of blocks representing the map

		Args:
			Map: 2D numpy array containing zeros and ones
			block_size: Map will be divided into block_size x block_size blocks
		"""
		super(BlockMap, self).__init__()

		self.block_size = block_size
		# expand map if not divisible by block size
		Map = self._expand_map(Map)
		self._map = Map

		h, w = Map.shape[0] // block_size, Map.shape[1] // block_size
		self.h, self.w = h, w
		self.blocks = [[None] * w for _ in range(h)]

		for i in range(h):
			for j in range(w):
				block_slice = Map[i * block_size: (i + 1) * block_size, j * block_size: (j + 1) * block_size]
				block_idx = list_to_block_idx(block_slice)
				self.blocks[i][j] = Block(block_idx, block_size, map_addr=(i, j))

	def __getitem__(self, key):
		y, x = key
		return self.blocks[y][x]

	def __setitem__(self, key, value):
		y, x = key
		self.blocks[y][x] = value

	def _expand_map(self, Map):
		"""Pad map with ones to make its h and w so that it is 
		evenly divisible into blocks.
		"""
		map_h, map_w = Map.shape
		rem_h, rem_w = map_h % self.block_size, map_w % self.block_size
		if rem_h > 0:
			Map = np.concatenate((Map, np.ones((rem_h, map_w))), axis=0)
		if rem_w > 0:
			Map = np.concatenate((Map, np.ones((map_h + rem_h, rem_w))), axis=1)
		return Map


	def block_neighbors(self, block):
		adjacent_blocks = self.adjacent_blocks(block.map_addr)
		valid_blocks = [(self[i], d) for i, d in adjacent_blocks if self._is_valid(*i)]
		return valid_blocks

	def adjacent_blocks(self, node):
		y, x = node
		dirs = [(0, -1), (-1, 0), (0, 1), (1, 0)]
		return [((y + d[0], x + d[1]), d) for d in dirs]

	def _is_valid(self, y, x):
		return (0 <= y < self.h) and (0 <= x < self.w)

	def get_node_block(self, node_addr):
		y, x = node_addr
		by, bx = y // self.block_size, x // self.block_size
		return self.blocks[by][bx], (y % self.block_size, x % self.block_size)

