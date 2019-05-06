from time import perf_counter
from collections import defaultdict

import numpy as np
from Block import *
from BlockMap import BlockMap
from LDDB import make_lddb
from visualizations import *
from Block_A_star import block_a_star
from A_star import a_star
from common import AttrDict, generate_random_map, l1_dist


np.random.seed(12)

##########################################################################
## Block A*
##########################################################################

block_size = 4

##################################
## for testing on specific maps
##################################
# Map = np.array([
# 	[0, 0, 0, 0,   0, 0, 0, 0],
# 	[0, 0, 0, 0,   0, 0, 0, 0],
# 	[0, 0, 0, 0,   0, 0, 0, 0],
# 	[0, 0, 0, 0,   0, 0, 0, 0],

# 	[0, 0, 0, 0,   0, 0, 0, 0],
# 	[0, 0, 0, 0,   0, 0, 0, 0],
# 	[0, 0, 0, 0,   0, 0, 0, 0],
# 	[0, 0, 0, 0,   0, 0, 0, 0],
	
# 	[0, 0, 0, 0,   0, 0, 0, 0],
# 	[0, 0, 0, 0,   0, 0, 0, 0],
# 	[0, 0, 0, 0,   0, 0, 0, 0],
# 	[0, 0, 0, 0,   0, 0, 0, 0],
# ])
# start = (1, 7)
# goal = (4, 2)
##################################


##################################
## for testing on random maps
##################################
Map, start, goal = generate_random_map(20, 20, p=0.3, start_and_goal=True)
##################################

block_map = BlockMap(Map, block_size)

lddb, pathsdb = make_lddb(block_size, from_file=True, save_to_file=True)
t1 = perf_counter()
goal_found, path = block_a_star(lddb, pathsdb, block_map, start, goal, l1_dist)
print(f'{perf_counter() - t1} sec')

print(start, goal)
print()
pretty_print(Map, block_size, path=path, start=start, goal=goal)
visualize_map(Map, start, goal, path)


##########################################################################
## A*
##########################################################################
t1 = perf_counter()
goal_found, path = a_star(Map, start, goal, l1_dist)
print(f'{perf_counter() - t1} sec')

pretty_print(Map, path=path, start=start, goal=goal)
visualize_map(Map, start, goal, path)
