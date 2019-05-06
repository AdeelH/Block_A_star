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


np.random.seed(1)

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
# 	[0, 0, 1, 0,   0, 0, 0, 0],
# 	[0, 1, 0, 1,   0, 0, 0, 0],

# 	[0, 1, 0, 1,   0, 0, 0, 0],
# 	[1, 1, 1, 1,   0, 0, 0, 0],
# 	[0, 0, 0, 0,   0, 0, 0, 0],
# 	[0, 0, 0, 0,   0, 0, 0, 0],

# 	[0, 0, 0, 1,   0, 0, 0, 0],
# 	[0, 0, 0, 1,   0, 0, 0, 0],
# 	[0, 0, 0, 1,   0, 0, 0, 0],
# 	[0, 0, 0, 1,   0, 0, 0, 0],
# ])
# start = (0, 0)
# goal = (11, 0)
##################################


##################################
## for testing on random maps
##################################
# for _ in range(50):
# 	Map, start, goal = generate_random_map(8, 8, p=0.4, start_and_goal=True)
Map, start, goal = generate_random_map(8, 8, p=0.4, start_and_goal=True)
pretty_print(Map, block_size, start=start, goal=goal)
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
