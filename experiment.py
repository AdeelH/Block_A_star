from time import perf_counter
from collections import defaultdict

import numpy as np
from BlockMap import BlockMap
from LDDB import make_lddb
from visualizations import *
from Block_A_star import block_a_star
from A_star import a_star
from common import AttrDict, generate_random_map, l1_dist


def experiment(algo, generate_map, data=AttrDict(), preprocessing=lambda m, d: m, maps=500, runs=100, ignore_failures=False):

	times = []
	failure_count = 0
	for i in range(maps):
		if i % 10 == 0:
			print(i)

		_m = generate_map()
		m = _m.copy()

		h, w = _m.shape

		starts = [(np.random.randint(h), np.random.randint(w)) for _ in range(runs)]
		goals = [(np.random.randint(h), np.random.randint(w)) for _ in range(runs)]

		for j, (start, goal) in enumerate(zip(starts, goals)):
			m = m | _m
			m[start[0], start[1]] = 0
			m[goal[0], goal[1]] = 0
			Map = preprocessing(m, data)

			t1 = perf_counter()
			goal_found, path = algo(Map, start, goal, data)
			times.append(perf_counter() - t1)
			if not goal_found:
				failure_count += 1
				if ignore_failures:
					continue

	return sum(times) / len(times), times, failure_count


def time_block_a_star(b_sz, h, w, p, maps, runs):

	lddb, pathsdb = make_lddb(b_sz, from_file=True, save_to_file=False)

	data = AttrDict({
		'block_size': b_sz,
		'lddb': lddb,
		'pathsdb': pathsdb,
		'h': l1_dist
	})
	generate_map = lambda: generate_random_map(h, w, p=p, start_and_goal=False)
	preprocessing = lambda m, d: BlockMap(m, d.block_size)
	algo = lambda m, s, g, d: block_a_star(d.lddb, d.pathsdb, m, s, g, d.h)

	avg_time, times, fc = experiment(algo, generate_map=generate_map, data=data, preprocessing=preprocessing, maps=maps, runs=runs)

	print('%e' % avg_time, fc)


def time_a_star(h, w, p, maps, runs):

	data = AttrDict({
		'h': l1_dist
	})
	generate_map = lambda: generate_random_map(h, w, p=p, start_and_goal=False)
	algo = lambda m, s, g, d: a_star(m, s, g, d.h)

	avg_time, times, fc = experiment(algo, generate_map=generate_map, data=data, maps=maps, runs=runs)

	print('%e' % avg_time, fc)

# Experiment parameters
np.random.seed(1)
h, w = 200, 200
maps = 50
runs = 50

for p in [0., .1, .2, .3, .4, .5]:
	print(p)
	np.random.seed(10)
	time_a_star(h=h, w=w, p=p, maps=maps, runs=runs)

for p in [0., .1, .2, .3, .4, .5]:
	print(p)
	np.random.seed(10)
	time_block_a_star(b_sz=4, h=h, w=w, p=p, maps=maps, runs=runs)
