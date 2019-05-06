import itertools
from heapq import heappush, heappop

# Adapted from:
# https://docs.python.org/3/library/heapq.html#priority-queue-implementation-notes
REMOVED = '<removed>'	  		# placeholder for a removed task
class PriorityQueue(object):

	def __init__(self):
		super(PriorityQueue, self).__init__()
		self.pq = []						 	# list of entries arranged in a heap
		self.entry_finder = {}			   	# mapping of items to entries
		self.counter = itertools.count()	 	# unique sequence count
		self.size = 0

	def push(self, item, priority=0):
		'Add a new item or update the priority of an existing item'
		if item in self.entry_finder:
			# if self.entry_finder[item][0] <= priority:
			# 	return
			self.remove(item)
		count = next(self.counter)
		entry = [priority, count, item]
		self.entry_finder[item] = entry
		heappush(self.pq, entry)
		self.size += 1

	def remove(self, item):
		'Mark an existing item as REMOVED.  Raise KeyError if not found.'
		entry = self.entry_finder.pop(item)
		entry[-1] = REMOVED
		self.size -= 1

	def pop(self):
		'Remove and return the lowest priority item. Raise KeyError if empty.'
		while self.pq:
			priority, count, item = heappop(self.pq)
			if item is not REMOVED:
				del self.entry_finder[item]
				self.size -= 1
				return item
		raise KeyError('pop from an empty priority queue')

	def top(self):
		while self.pq:
			entry = heappop(self.pq)
			if entry[-1] is not REMOVED:
				heappush(self.pq, entry)
				return entry[-1], entry[0]
		raise KeyError('top() on an empty priority queue')

	def priority(self, item):
		return self.entry_finder[item][0]


	def __len__(self):
		return self.size

	def __contains__(self, item):
		return item in self.entry_finder

	def __str__(self):
		return str(self.pq)
