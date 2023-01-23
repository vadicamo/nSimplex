import time
import unittest

from scipy.spatial.distance import squareform
import torch

from nsimplex_torch import NSimplex


class Test_NSimplexTorch(unittest.TestCase):

    def _test_single(self, dim, n_pivots, device, seed):
        torch.manual_seed(seed)

        pivots = torch.rand((n_pivots, dim)).to(device)
        o1 = torch.rand((dim,)).to(device)
        o2 = torch.rand((dim,)).to(device)

        # pivot-pivot distance matrix with shape (n_pivots, n_pivots)
        pp = torch.pdist(pivots)
        pp = torch.from_numpy(squareform(pp.cpu())).to(device)

        # groundtruth distance:
        oo = torch.norm(o1 - o2, p=2)

        # objects-pivots distances with shape (n_pivots,)
        o1p = torch.cdist(o1[None, :], pivots)[0]
        o2p = torch.cdist(o2[None, :], pivots)[0]

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

    def _test_batched(self, dim, n_objects, n_pivots, device, seed):
        pivots = torch.rand((n_pivots, dim)).to(device)
        o1 = torch.rand((n_objects, dim)).to(device)
        o2 = torch.rand((n_objects, dim)).to(device)

        # pivot-pivot distance matrix with shape (n_pivots, n_pivots)
        pp = torch.pdist(pivots)
        pp = torch.from_numpy(squareform(pp.cpu())).to(device)

        # groundtruth distance:
        oo = torch.nn.functional.pairwise_distance(o1, o2)

        # objects-pivots distances with shape (n_objects, n_pivots)
        o1p = torch.cdist(o1, pivots)
        o2p = torch.cdist(o2, pivots)

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

    def test_single(self):
        dim = 1024
        n_pivots = 256
        seed = 7

        for device in ('cpu', 'cuda'):
            with self.subTest(device=device):
                device = torch.device(device)
                self._test_single(dim, n_pivots, device, seed)

    def test_batched(self):
        dim = 1024
        n_objects = 64
        n_pivots = 256
        seed = 7

        for device in ('cpu', 'cuda'):
            with self.subTest(device=device):
                device = torch.device(device)
                self._test_batched(dim, n_objects, n_pivots, device, seed)


if __name__ == '__main__':
    unittest.main()