import numpy as np


# convenience class, allows dot access for dicts
# source: https://stackoverflow.com/a/14620633/5908685
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def get_path_from_parent_map(parent_map, end):

	path = []
	curr = end

	while curr is not None:
		path.append(curr)
		curr = parent_map[curr]
	return path[::-1]

def is_visitable(container, node):
	y, x = node
	h, w = container.shape
	return 0 <= y < h and 0 <= x < w and container[y, x] == 0


def visitable(container, nodes):
	return [n for n in nodes if is_visitable(container, n)]


def generate_random_map(h, w, p=0.2, start_and_goal=True):
	m = ((np.random.rand(h, w)) < p)
	if start_and_goal:
		start = (np.random.randint(h), np.random.randint(w))
		goal = (np.random.randint(h), np.random.randint(w))
		# make sure no obstacle in start and goal cells
		m[start[0], start[1]] = 0
		m[goal[0], goal[1]] = 0

		return m, start, goal
	return m


def l1_dist(node, goal):
	y, x = node
	gy, gx = goal
	return np.abs(y - gy) + np.abs(x - gx)
