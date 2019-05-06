from time import perf_counter
from collections import defaultdict

import numpy as np
from Block import Block, list_to_block_idx
from PriorityQueue import PriorityQueue


class BlockMap(object):
	def __init__(self, Map, block_size):
		super(BlockMap, self).__init__()

		h, w = Map.shape[0] // block_size, Map.shape[1] // block_size
		self.h, self.w = h, w
		self.block_size = block_size
		self.blocks = [[None] * w for _ in range(h)]
		self._map = Map

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

