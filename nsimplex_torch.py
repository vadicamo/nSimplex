import warnings

import torch
from tqdm import trange


class NSimplex (object):

    def __init__(self):
        self._base = None

    @staticmethod
    def atleast_2d(x):
        """ Ensures that the output is at least 2-dimensional. See numpy.atleast_2d(). """
        while x.ndim < 2:
            x = x.unsqueeze(0)
        return x

    @classmethod
    def _get_apex(cls, base, distances):
        """ Find new apices given a `base` and the `distances` to pivots.
        This function works on batches of inputs.

        Args:
            base (torch.tensor): a (P,P-1)-shaped tensor representing the simplex base.
            distances (torch.tensor): a (N,P)-shaped tensor containing distances to the P pivots for N objects.

        Returns:
            torch.tensor: a (N,P)-shaped tensor containing the new apices.
        """
        distances = cls.atleast_2d(distances)

        assert base.shape[0] == distances.shape[1], \
            f'Base size and number of distances should match, found {base.shape[0]} and {distances.shape[1]})'

        b, n = distances.shape
        apex = torch.zeros((b, n), dtype=distances.dtype, device=distances.device)
        apex[:, 0] = distances[:, 0]

        for k in range(1, n):
            ldist = ((apex[:, :k] - base[k, :k]) ** 2).sum(axis=1)
            diff = distances[:, k] ** 2 - ldist

            x_n = base[k, k - 1]
            y_n = apex[:, k - 1]
            w = y_n - diff / (2 * x_n)
            z = (y_n ** 2 - w ** 2).sqrt()

            is_significant = torch.isfinite(z)
            if not is_significant.all():
                warnings.warn(f"one or more points does not satisfies the n-point property, discarding pivot {k}")

            # fallback to w = y_n and z = 0 for degenerate cases
            apex[:, k - 1] = torch.where(is_significant, w, y_n)
            # apex[:, k] = torch.where(is_significant, z, 0) torch.where fails with python scalars
            apex[is_significant, k] = z[is_significant]

        return apex

    def build_base(self, distances, progress):
        """ Builds the simplex base given the `distances` between pivots.

        Args:
            distances (torch.tensor): a (P,P)-shaped tensor containing distances between pivots.
            progress (bool, optional): if True, show progress with tqdm. Defaults to False.

        TODO: change to accept also (½xPx(P-1))-shaped array containing linearized values of the
        upper triangular part (diagonal excluded) of the distance matrix.
        See docs of torch.nn.functional.pdist().
        """
        n = distances.shape[0]
        base = torch.zeros((n, n - 1), dtype=distances.dtype, device=distances.device)

        base[1, 0] = distances[1, 0]
        for k in trange(2, n, disable=not progress):
            base[k, :k] = self._get_apex(base[:k, :k-1], distances[k, :k])

        self._base = base

    def embed(self, distances):
        """ Embed objects.

        Args:
            distances (torch.tensor): a (P,)- or (N,P)-shaped tensor containing the distance
                                      to pivots of the objects to embed.

        Returns:
            torch.tensor: a (N,P)-shaped tensor containing the embedding. N=1 if input was an 1D-tensor.
        """
        assert self._base is not None, "Simplex base is not built, call build_base() first"
        return self._get_apex(self._base, distances)

    def estimate(self, x, y, kind='zenit'):
        """ Generate distance estimations from embeddings `x` and `y`.

        Args:
            x (torch.tensor): a (P,)- or (N,P)-shaped tensor of embeddings
            y (torch.tensor): a (P,)- or (N,P)-shaped tensor of embeddings
            kind (str, optional): Kind of estimation. Can be one of 'lower', 'upper', or 'zenit'. Defaults to 'zenit'.

        Returns:
            torch.tensor: a (N,)-shaped tensor containing distance estimantes.
        """
        assert kind in ('lower', 'upper', 'zenit'), f'Invalid estimate: {kind}'

        x = self.atleast_2d(x)
        y = self.atleast_2d(y)

        # if kind == 'lower':
        #     pass

        if kind == 'upper':
            x = torch.cat((x[:, :-1], - x[:, -1:]), dim=1)

        elif kind == 'zenit':
            z = torch.zeros_like(x[:, :1])
            x = torch.cat((x[:, :-1], z, x[:, -1:]), dim=1)
            y = torch.cat((y, z), dim=1)

        return torch.nn.functional.pairwise_distance(x, y)
