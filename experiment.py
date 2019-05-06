from time import perf_counter
from collections import defaultdict

import numpy as np
from BlockMap import BlockMap
from LDDB import make_lddb
from visualizations import *
from Block_A_star import block_a_star
from A_star import a_star
from common import AttrDict, generate_random_map, l1_dist


def experiment(algo, generate_map, data=AttrDict(), preprocessing=lambda m, d: m, runs=100, ignore_failures=True):

	times = []
	failure_count = 0
	while len(times) != runs:
		_m = generate_map()
		Map, start, goal = preprocessing(_m, data)

		t1 = perf_counter()
		goal_found, path = algo(Map, start, goal, data)
		if not goal_found:
			failure_count += 1
			if ignore_failures:
				continue
		times.append(perf_counter() - t1)
		# print(len(times))

	return sum(times) / runs, times, failure_count


def time_block_a_star(b_sz, h, w, p, runs):

	lddb, pathsdb = make_lddb(b_sz, from_file=True, save_to_file=True)

	data = AttrDict({
		'block_size': b_sz,
		'lddb': lddb,
		'pathsdb': pathsdb,
		'h': l1_dist
	})
	generate_map = lambda: generate_random_map(h, w, p=p)
	preprocessing = lambda m, d: (BlockMap(m[0], d.block_size), m[1], m[2])
	algo = lambda m, s, g, d: block_a_star(d.lddb, d.pathsdb, m, s, g, d.h)

	avg_time, _, fc = experiment(algo, generate_map=generate_map, data=data, preprocessing=preprocessing, runs=runs)

	print('%e' % avg_time, fc)


def time_a_star(h, w, p, runs):

	data = AttrDict({
		'h': l1_dist
	})
	generate_map = lambda: generate_random_map(h, w, p=p)
	algo = lambda m, s, g, d: a_star(m, s, g, d.h)

	avg_time, _, fc = experiment(algo, generate_map=generate_map, data=data, runs=runs)

	print('%e' % avg_time, fc)


np.random.seed(1)
h, w = 8, 8
p = 0.4
runs = 1
time_block_a_star(b_sz=4, h=h, w=w, p=p, runs=runs)
time_a_star(h=h, w=w, p=p, runs=runs)
