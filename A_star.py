from functools import lru_cache
from PriorityQueue import PriorityQueue
from common import get_path_from_parent_map, visitable


def _neighbors(Map, node):
	return visitable(Map, _adjacent_nodes(node))


def _adjacent_nodes(node):
	y, x = node
	return [(y, x - 1), (y - 1, x), (y, x + 1), (y + 1, x)]


def a_star(Map, start, goal, h):
	frontier = PriorityQueue()
	parent_map = {}
	g = {}

	goal_found = False
	g[start] = 0
	parent_map[start] = None
	frontier.push(start, 0)

	while len(frontier) > 0:
		current = frontier.pop()
		if current == goal:
			goal_found = True
			break

		for nghb in _neighbors(Map, current):
			new_g = g[current] + 1
			if nghb not in g or new_g < g[nghb]:
				g[nghb] = new_g
				priority = new_g + h(nghb, goal)
				frontier.push(nghb, priority)
				parent_map[nghb] = current

	if goal_found:
		return True, get_path_from_parent_map(parent_map, goal)
	return False, []

