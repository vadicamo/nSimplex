import time
import unittest

import numpy as np
from scipy.spatial.distance import pdist, cdist, euclidean, squareform

from nsimplex import NSimplex


class Test_NSimplexNumpy(unittest.TestCase):

    def test_single(self):
        n_pivots = 256
        dim = 1024

        rng = np.random.default_rng(7)

        pivots = rng.random((n_pivots, dim))
        o1 = rng.random((dim,))
        o2 = rng.random((dim,))

        # pivot-pivot distance matrix with shape (n_pivots, n_pivots)
        pp = squareform(pdist(pivots))

        # groundtruth distance:
        oo = euclidean(o1, o2)

        # objects-pivots distances with shape (n_pivots,)
        o1p = cdist(o1[None, :], pivots)[0]
        o2p = cdist(o2[None, :], pivots)[0]

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

        self.assertTrue(lwb <= oo  <= upb)
        self.assertTrue(lwb <= zen <= upb)
        print(f'l: {lwb:.5f} <= o: {oo:.5f} (m: {(lwb+upb) / 2:.5f} - z: {zen:.5f}) <= u: {upb:.5f}')

    def test_batched(self):
        n_pivots = 256
        n_objects = 64
        dim = 1024

        rng = np.random.default_rng(7)

        pivots = rng.random((n_pivots, dim))
        o1 = rng.random((n_objects, dim))
        o2 = rng.random((n_objects, dim))

        # pivot-pivot distance matrix with shape (n_pivots, n_pivots)
        pp = squareform(pdist(pivots))

        # groundtruth distance:
        oo = np.sqrt(((o1 - o2) ** 2).sum(axis=1))

        # objects-pivots distances with shape (n_objects, n_pivots)
        o1p = cdist(o1, pivots)
        o2p = cdist(o2, pivots)

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

        self.assertTrue((lwb <= oo ).all() and (oo  <= upb).all())
        self.assertTrue((lwb <= zen).all() and (zen <= upb).all())

        mean = (lwb + upb) / 2
        for l, o, m, z, u in zip(lwb, oo, mean, zen, upb):
            print(f'l: {l:.5f} <= o: {o:.5f} (m: {m:.5f} - z: {z:.5f}) <= u: {u:.5f}')


if __name__ == '__main__':
    unittest.main()