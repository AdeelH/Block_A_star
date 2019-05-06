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
	while len(state.pq) > 0 and state.heapvalue[state.pq.top()[0]] < length:
		curr_block = state.pq.pop()
		ingress = get_ingress_nodes(state, curr_block)
		if len(ingress) == 0:
			continue
		if curr_block.map_addr == goal_block.map_addr:
			path_lens = [state.g[curr_block][y] + state.lddb[curr_block.idx].get((y, goal_block_node), np.inf) for y in ingress]
			min_y_to_goal = np.min(path_lens)
			best_y = ingress[np.argmin(path_lens)]
			if goal_block_node != best_y:
				state.parent[(goal_block, goal_block_node)] = (curr_block, best_y)
			if min_y_to_goal < np.inf:
				length = min(length, min_y_to_goal)
			else:
				state.heapvalue[curr_block] = np.inf
		expand_block(state, curr_block, ingress, h)
	if length < np.inf:
		return True, recover_path(state, goal_block, goal_block_node)
	else:
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

def expand_block(state, curr_block, ingress_nodes, h):
	# print(f'>> curr_block: {curr_block.map_addr}')
	# print(f'>>>> ingress_nodes: {ingress_nodes}')
	g, g_changed, lddb = state.g, state.g_changed, state.lddb

	# neighboring blocks
	nbs = state.Map.block_neighbors(curr_block)
	# for each neighboring block, next_block
	for next_block, direction in nbs:
		# get valid egress nodes on that side
		egress_nodes = get_egress_nodes(state, curr_block, next_block, direction)
		if len(egress_nodes) == 0:
			continue
		# for each valid egress node, e, and its neighbor in next_block, e_nb
		for e, e_nb in egress_nodes:

			e_old_g = state.g[curr_block].get(e, np.inf)

			# g values for x through each ingress cell
			gs_to_e = [
				g[curr_block][y] + lddb[curr_block.idx].get((y, e), np.inf) for y in ingress_nodes
			]
			# best (min) g value
			e_new_g = np.min(gs_to_e)

			g[curr_block][e] = min(e_old_g, e_new_g)
			g_changed[curr_block][e] = False

			# if g value has changed, set that ingress node as e's parent
			if e_new_g < e_old_g:
				nearest_ingress_node = ingress_nodes[np.argmin(gs_to_e)]
				state.parent[(curr_block, e)] = (curr_block, nearest_ingress_node)

			e_nb_old_g = g[next_block].get(e_nb, np.inf)
			e_nb_new_g = g[curr_block][e] + 1
			
			g[next_block][e_nb] = min(e_nb_old_g, e_nb_new_g)
			# if g value has changed, 
			if e_nb_new_g < e_nb_old_g:
				# set e as e_nb's parent 
				state.parent[(next_block, e_nb)] = (curr_block, e)
				# and mark e_nb as a possible ingress node for next_block
				g_changed[next_block][e_nb] = True

		# g + h values for next_block via each possible egress node
		dists_to_next_block = [
			g[next_block][e_nb] + state.h(next_block, e_nb) for _, e_nb in egress_nodes
		]
		# best g + h
		new_priority = np.min(dists_to_next_block)
		# if improved, push next_block on to the heap
		if new_priority < state.heapvalue.get(next_block, np.inf) or any(g_changed[next_block].get(e_nb, False) for _, e_nb in egress_nodes):
			state.heapvalue[next_block] = new_priority
			state.pq.push(next_block, new_priority)
			e, x_nb = egress_nodes[np.argmin(dists_to_next_block)]



###############################################################################################
# Helpers
###############################################################################################
def init(state, block, node):
	dists, parent_map = bfs_to_all_points(block, node)
	for k, v in dists.items():
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
		p_block, p_node = state.parent[(curr_block, curr_node)]

		if p_block is None:
			break

		if p_block == curr_block:
			in_block_path = state.pathsdb[curr_block.idx][(curr_node, p_node)][1:]
			path.extend([to_global_node(curr_block, n) for n in in_block_path])
		else:
			path.append(to_global_node(p_block, p_node))

		curr_block, curr_node = p_block, p_node
	return path



