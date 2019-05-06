from collections import deque
from functools import lru_cache

from common import is_visitable, visitable


class InvalidNodeError(Exception):
	pass


class Block(object):
	def __init__(self, idx, size, map_addr=None):
		super(Block, self).__init__()
		assert idx < 2**(size**2)
		self.idx = idx
		self.size = size
		self.shape = (size, size)
		self.map_addr = map_addr

	def __len__(self):
		return self.size

	def __getitem__(self, node):
		y, x = node
		pos = y * self.size + x
		if not self._is_valid(y, x):
			raise InvalidNodeError(f"({y}, {x})")
		return (self.idx & (1 << pos)) >> pos

	def __contains__(self, node):
		y, x = node
		return self._is_valid(y, x)

	def __str__(self):
		out = [None] * self.size
		for i in range(self.size):
			out[i] = str([self.__getitem__((i, j)) for j in range(self.size)])
		return f"idx: {self.idx}\n" + "\n".join(out)

	def _is_valid(self, y, x):
		return (0 <= y < self.size) and (0 <= x < self.size)


def neighbors(block, node):
	return visitable(block, adjacent_nodes(node))


@lru_cache(32)
def adjacent_nodes(node):
	y, x = node
	return [(y, x - 1), (y - 1, x), (y, x + 1), (y + 1, x)]


def boundary_nodes(block):
	b = len(block)
	top    = [(0    , x    ) for x in range(b       )]
	bottom = [(b - 1, x    ) for x in range(b       )]
	right  = [(y    , b - 1) for y in range(1, b - 1)]
	left   = [(y    , 0    ) for y in range(1, b - 1)]
	return top + bottom + left + right


def is_boundary_node(block, node):
	y, x = node
	b = len(block) - 1
	return y == 0 or x == 0 or y == b or x == b


def list_to_block_idx(ls):
	idx = 0
	idx_ptr = 1
	size = len(ls)
	for i in range(size):
		for j in range(size):
			if ls[i][j] != 0:
				idx |= idx_ptr
			idx_ptr <<= 1
	return idx


def list_to_block(ls):
	return Block(list_to_block_idx(ls), size)

