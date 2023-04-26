import time
import unittest

import numpy as np
from scipy.spatial.distance import pdist, cdist, euclidean, squareform

from nsimplex import NSimplex


class Test_NSimplexNumpy(unittest.TestCase):

    def test_single(self):
        n_pivots = 256
        dim = 1024
        metric='euclidean'
        print(f"test_single with  {n_pivots} pivots and {dim} dimensions and metric {metric}")

        rng = np.random.default_rng(7)

        pivots = rng.random((n_pivots, dim))
        o1 = rng.random((dim,))
        o2 = rng.random((dim,))

        # pivot-pivot distance matrix with shape (n_pivots, n_pivots)
        pp = squareform(pdist(pivots,metric=metric))

        # groundtruth distance:
        oo =cdist(o1[None, :],o2[None, :],metric=metric)[0,0] 

        # objects-pivots distances with shape (n_pivots,)
        o1p = cdist(o1[None, :], pivots,metric=metric)[0]
        o2p = cdist(o2[None, :], pivots,metric=metric)[0]

        tic = time.time()
        simplex = NSimplex()  # show progress while building base
        simplex.build_base(pp, progress=True)
        print(f'Build time: {time.time() - tic} s')

        tic = time.time()
        proj_o1 = simplex.embed(o1p)  # shape (n_pivots,)
        proj_o2 = simplex.embed(o2p)  # shape (n_pivots,)
        print(f'Project time: {time.time() - tic} s')

        # estimate bounds
        lwb = simplex.estimate(proj_o1, proj_o2, kind='lower')[0]  # scalar
        upb = simplex.estimate(proj_o1, proj_o2, kind='upper')[0]  # scalar
        zen = simplex.estimate(proj_o1, proj_o2, kind='zenit')[0]  # scalar

        print(f'lwb: {lwb:.5f} <= original_dist: {oo:.5f} (mean: {(lwb+upb) / 2:.5f} - zen: {zen:.5f}) <= upb: {upb:.5f}')
        
        self.assertTrue(lwb <= oo  <= upb)
        self.assertTrue(lwb <= zen <= upb)

    def test_batched(self):
        
        n_pivots = 256
        n_objects = 10
        dim = 1024
        metric='euclidean'
        print(f"test_batched- with batch size {n_objects} and {n_pivots} pivots and {dim} dimensions and metric {metric}")

        rng = np.random.default_rng(7)

        pivots = rng.random((n_pivots, dim))
        o1 = rng.random((n_objects, dim))
        o2 = rng.random((n_objects, dim))

        # pivot-pivot distance matrix with shape (n_pivots, n_pivots)
        pp = squareform(pdist(pivots, metric=metric))

        # groundtruth distance:
        oo = cdist(o1,o2,metric=metric).diagonal()
  
        # objects-pivots distances with shape (n_objects, n_pivots)
        o1p = cdist(o1, pivots,metric=metric)
        o2p = cdist(o2, pivots,metric=metric)

        tic = time.time()
        simplex = NSimplex()  # show progress while building base
        simplex.build_base(pp, progress=True)
        print(f'Build time: {time.time() - tic} s')

        tic = time.time()
        proj_o1 = simplex.embed(o1p)  # shape (n_objects, n_pivots)
        proj_o2 = simplex.embed(o2p)  # shape (n_objects, n_pivots)
        print(f'Project time: {time.time() - tic} s')

        # estimate bounds
        lwb = simplex.estimate(proj_o1, proj_o2, kind='lower')  # shape (n_objects,)
        upb = simplex.estimate(proj_o1, proj_o2, kind='upper')  # shape (n_objects,)
        zen = simplex.estimate(proj_o1, proj_o2, kind='zenit')  # shape (n_objects,)

        mean = (lwb + upb) / 2
        for l, original_dist, m, z, u in zip(lwb, oo, mean, zen, upb):
            print(f'lwb: {l:.5f} <= original_dist: {original_dist:.5f} (mean: {m:.5f} - zen: {z:.5f}) <= upb: {u:.5f}')
       
        self.assertTrue((lwb <= oo ).all() and (oo  <= upb).all())
        self.assertTrue((lwb <= zen).all() and (zen <= upb).all())


if __name__ == '__main__':
    unittest.main()