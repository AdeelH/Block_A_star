from time import perf_counter
from collections import defaultdict

import numpy as np
from Block import boundary_nodes
from PriorityQueue import PriorityQueue
from LDDB import bfs_to_all_points
from visualizations import *
from common import AttrDict, get_path_from_parent_map, is_visitable, visitable


# As descibed in the paper:
# 
# Algorithm 2 Block A*
#
# PROC: Block A* (LDDB, start, goal):
#	 startBlock = init(start)
#	 goalBlock = init(goal)
#	 length = ∞
#	 insert startBlock into OPEN
#	 while (OPEN ̸= empty) and ((OPEN.top).heapvalue < length) do
#	 	curBlock = OPEN.pop
#	 	Y = set of all curBlock’s ingress nodes
#	 	if curBlock == goalBlock then
#	 		length = min_y∈Y ( y.g+dist(y, goal), length)
#	 	end if
#	 	Expand( curBlock, Y )
#	 end while
#	 if length ̸= ∞ then
#	 	Reconstruct solution path
#	 else
#	 	return Failure
#	 end if

def block_a_star(lddb, pathsdb, Map, start, goal, h):
	state = AttrDict({
		'Map': Map,
		'start': start,
		'goal': goal,
		'g': defaultdict(dict),
		'g_changed': defaultdict(dict),
		'heapvalue': {},
		'pq': PriorityQueue(),
		'parent': {},
		'lddb': lddb,
		'pathsdb': pathsdb,
		'h': lambda block, node: h(to_global_node(block, node), goal),
		'start': start
	})

	start_block, start_block_node = Map.get_node_block(start)
	goal_block , goal_block_node  = Map.get_node_block(goal)

	init(state, start_block, start_block_node)
	init(state, goal_block, goal_block_node)

	state.g[start_block][start_block_node] = 0
	state.g_changed[start_block][start_block_node] = True
	state.parent[(start_block, start_block_node)] = (None, None)

	state.pq.push(start_block, 0)
	state.heapvalue[start_block] = 0

	length = np.inf
	while len(state.pq) > 0 and state.pq.top()[1] < length:
		curr_block = state.pq.pop()
		# print('\n\n>>>>>> curr_block: ', curr_block.map_addr)
		ingress = get_ingress_nodes(state, curr_block)
		# print('ingress: ', ingress)
		if len(ingress) == 0:
			continue
		if curr_block.map_addr == goal_block.map_addr:
			path_lens = [state.g[curr_block][y] + state.lddb[curr_block.idx].get((y, goal_block_node), np.inf) for y in ingress]
			min_y_to_goal = np.min(path_lens)
			best_y = ingress[np.argmin(path_lens)]
			if goal_block_node != best_y:
				# print(f'parent[{(goal_block.map_addr, goal_block_node)}] = {(curr_block.map_addr, best_y)}')
				state.parent[(goal_block, goal_block_node)] = (curr_block, best_y)
			if min_y_to_goal < np.inf:
				length = min(length, min_y_to_goal)
			else:
				state.heapvalue[curr_block] = np.inf
		expand_block(state, curr_block, ingress, h)
		# print('pq: ', state.pq)
	if length < np.inf:
		# print(f'\nGoal found! Length = {length}\n')
		return True, recover_path(state, goal_block, goal_block_node)
	else:
		# print('Goal not found!')
		return False, []


# As descibed in the paper:
# 
# Algorithm 1 Expand curBlock. Y is the set of curBlock’s
# ingress cells.
# 
# Expand(curBlock, Y ):
# 	for side of curBlock with neighbor nextBlock do
# 		for valid egress node x on current side do
# 			x′ = egress neighbor of x on current side
# 			x.g = miny∈Y (y.g + LDDB(y, x), x.g)
# 			x′.g = min(x′.g, x.g + cost(x, x′))
# 		end for
# 		newheapvalue = min_updated_x′ (x′.g + x′.h)
# 		if newheapvalue < nextBlock.heapvalue then
# 			nextBlock.heapvalue = newheapvalue
# 			if nextBlock not in OPEN then
# 				insert nextBlock into OPEN
# 			else
# 				UpdateOPEN(nextBlock)
# 			end if
# 		end if
# 	end for
# 

def expand_block(state, curr_block, Y, h):

	nghbs = state.Map.neighbors(curr_block)
	for next_block, direction in nghbs:
		# print(f'\nnext_block: {next_block.map_addr}')
		xs = get_egress_nodes(state, curr_block, next_block, direction)
		# print(f'egress: {[x for x, _ in xs]}')
		if len(xs) == 0:
			continue

		for x, x_nghb in xs:
			# print(f'x: {x}, x_nghb: {x_nghb}')
			x_old_g = state.g[curr_block].get(x, np.inf)
			path_lens = [state.g[curr_block][y] + state.lddb[curr_block.idx].get((y, x), np.inf) for y in Y]
			x_new_g = np.min(path_lens)
			y = Y[np.argmin(path_lens)]

			if x_new_g < x_old_g:
				state.parent[(curr_block, x)] = (curr_block, y)
				# print(f'parent[{(curr_block.map_addr, x)}] = {(curr_block.map_addr, y)}')

			state.g[curr_block][x] = min(x_old_g, x_new_g)
			state.g_changed[curr_block][x] = False

			x_nghb_old_g = state.g[next_block].get(x_nghb, np.inf)
			x_nghb_new_g = state.g[curr_block][x] + 1
			
			state.g[next_block][x_nghb] = min(x_nghb_old_g, x_nghb_new_g)
			if x_nghb_new_g < x_nghb_old_g:
				state.g_changed[next_block][x_nghb] = True
				state.parent[(next_block, x_nghb)] = (curr_block, x)
				# print(f'parent[{(next_block.map_addr, x_nghb)}] = {(curr_block.map_addr, x)}')

			# print(f'g[{next_block.map_addr}][{x_nghb}]: {state.g[next_block][x_nghb]}')

		path_lens = [state.g[next_block][x_nghb] + state.h(next_block, x_nghb) for _, x_nghb in xs]
		new_priority = np.min(path_lens)
		# print(path_lens, new_priority, state.heapvalue.get(next_block, np.inf))
		if new_priority < state.heapvalue.get(next_block, np.inf):
			state.heapvalue[next_block] = new_priority
			state.pq.push(next_block, new_priority)
			x, x_nghb = xs[np.argmin(path_lens)]
			# print(f'parent[{(next_block.map_addr, x_nghb)}] = {(curr_block.map_addr, x)}')
			# state.parent[(next_block, x_nghb)] = (curr_block, x)
			# print(f'new_priority: {next_block.map_addr}, {new_priority}')





###############################################################################################
# Helpers
###############################################################################################
def init(state, block, node):
	# print(f'block {block}')
	dists, parent_map = bfs_to_all_points(block, node)
	# print(f'node {node}')
	for k, v in dists.items():
		if is_boundary_node(block, k):
			# print(f'k {k}')
			state.lddb[block.idx][(node, k)] = v
			state.lddb[block.idx][(k, node)] = v
			p = get_path_from_parent_map(parent_map, k)
			state.pathsdb[block.idx][(node, k)] = p
			state.pathsdb[block.idx][(k, node)] = p[::-1]


def get_egress_nodes(state, curr_block, next_block, direction):
	dy, dx = direction
	boundary = visitable(curr_block, boundary_nodes(curr_block))
	sz = len(curr_block)
	if dy == 0:
		x = 0 if dx == -1 else len(curr_block) - 1
		next_x = (x + dx) % sz
		egress = [b for b in boundary if b[1] == x]
		egress_nghb = [(by, next_x) for by, bx in boundary if ((bx + dx) % sz) == next_x]
	else:
		y = 0 if dy == -1 else len(curr_block) - 1
		next_y = (y + dy) % sz
		egress = [b for b in boundary if b[0] == y]
		egress_nghb = [(next_y, bx) for by, bx in boundary if ((by + dy) % sz) == next_y]
	return [(e, en) for e, en in zip(egress, egress_nghb) if is_visitable(next_block, en)]


def get_ingress_nodes(state, block):
	block_g = state.g[block]
	block_g_flag = state.g_changed[block]
	return [node for node in block_g.keys() if block_g_flag.get(node, False)]


def to_global_node(block, node):
	by, bx = block.map_addr
	sz = block.size
	y, x = node
	return (by * sz + y), (bx * sz + x)

def recover_path(state, goal_block, goal_block_node):
	path = [to_global_node(goal_block, goal_block_node)]
	curr_block, curr_node = goal_block, goal_block_node
	while True:
		try:
			p_block, p_node = state.parent[(curr_block, curr_node)]
		except:
			print('Error!')
			print(path)
			print((curr_block.map_addr, curr_node))
			print(state.parent)
			break

		if p_block is None:
			break

		if to_global_node(p_block, p_node) in path:
			print(path)
			print(to_global_node(p_block, p_node))
			pretty_print(state.Map._map, state.Map.block_size, path=path, start=state.start, goal=path[0])
			break

		if p_block == curr_block:
			in_block_path = state.pathsdb[curr_block.idx][(curr_node, p_node)][1:]
			path.extend([to_global_node(curr_block, n) for n in in_block_path])
		else:
			path.append(to_global_node(p_block, p_node))

		curr_block, curr_node = p_block, p_node
	return path



