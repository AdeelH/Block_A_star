from collections import deque
from Block import *
from time import perf_counter as timer
import pickle

from common import get_path_from_parent_map
from Block import neighbors


def bfs_to_all_points(block, start):
	frontier = deque()
	frontier.append(start)
	dists = {start: 0}
	parent = {}
	parent[start] = None

	while len(frontier) > 0:
		curr = frontier.popleft()
		nghbs = neighbors(block, curr)

		node_dist_from_curr = dists[curr] + 1
		for n in nghbs:
			if n not in dists:# or node_dist_from_curr < dists[n]:
				dists[n] = node_dist_from_curr
				frontier.append(n)
				parent[n] = curr

	return dists, parent


def make_lddb(block_size, from_file=False, save_to_file=True):

	if from_file:
		with open('lddb.pkl', 'rb') as f_in:
			lddb, paths = pickle.load(f_in)
		return lddb, paths

	t1 = timer() 
	size = block_size
	lddb = {}
	paths = {}
	for idx in range(2**(size**2)):
		block = Block(idx, size)
		block_dists = {}
		block_paths = {}
		for n in visitable(block, boundary_nodes(block)):
			start = n
			dists, parent_map = bfs_to_all_points(block, start)
			for k, v in dists.items():
				if is_boundary_node(block, k):
					block_dists[(start, k)] = v
					block_paths[(start, k)] = get_path_from_parent_map(parent_map, k)

		lddb[idx] = block_dists
		paths[idx] = block_paths

	print()
	print(f"lddb created. Size:  {len(lddb)}. {timer() - t1} seconds")
	print()

	if not from_file and save_to_file:
		# save to file
		f_pkl = 'lddb.pkl'
		data = (lddb, paths)
		with open(f_pkl, "wb") as f:
		    pickle.dump(data, f)

	return lddb, paths
	