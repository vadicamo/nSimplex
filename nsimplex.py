import warnings

import numpy as np
from tqdm import trange


class NSimplex (object):

    def __init__(self):
        self._base = None

    @classmethod
    def _get_apex(cls, base, distances):
        """ Find new apices given the `base` simplex generated by the pivots {p1, .., pN} and the `distances` of the input object to the pivots {p1, .., pN}.
        This function works on batches of inputs.

        Args:
            base (numpy.ndarray): a (N,N-1)-shaped traingular array representing the simplex base. The rows are the coordinates of the vertices of the simplex built from a set of N pivots. The first vertex is always the origin. 
            distances (numpy.ndarray): a (B,N)-shaped array containing distances to the N pivots for B objects.

        Returns:
            numpy.ndarray: a (B,N)-shaped array containing the Euclidean coordinates of the new apices.
        """
        distances = np.atleast_2d(distances)
        #distances = distances.astype(np.double)

        assert base.shape[0] == distances.shape[1], \
            f'Base size and number of distances should match, found {base.shape[0]} and {distances.shape[1]})'

        b, n = distances.shape
        apex = np.zeros((b, n), dtype=distances.dtype)
        apex[:, 0] = distances[:, 0]

        for k in range(1, n):
            ldist = ((apex[:, :k] - base[k, :k]) ** 2).sum(axis=1) #squared Euclidean distance between new apex and k-th vertex of the base
            diff = distances[:, k] ** 2 - ldist

            x_n = base[k, k - 1]
            y_n = apex[:, k - 1] 
            w = y_n - diff / (2 * x_n) #second last coordinate of the new apex
            z = np.sqrt(y_n ** 2 - w ** 2)

            is_significant = np.isfinite(z) #& np.isreal(z)

            if not is_significant.all():
                warnings.warn(f"one or more points does not satisfies the n-point property, or the max dim was reached (diff for vertex-pivot {k} is {diff})")

            # fallback to w = y_n and z = 0 for degenerate cases
            apex[:, k - 1] = np.where(is_significant, w, y_n)
            apex[:, k] = np.where(is_significant, z, 0)

        return apex

    def build_base(self, distances, progress=False):
        """ Builds the (n-1)-dimensional simplex base given the `distances` between n pivots {p1, .., pn}. The base simplex is represented as a n x (n-1) matrix. The rows are the Euclidean coordinates of the vertices of the simplex.

        Args:
            distances (numpy.ndarray): a (n,n)-shaped array containing distances between pivots.
            progress (bool, optional): if True, show progress with tqdm. Defaults to False.

        TODO: change to accept also (½ n x (n-1))-shaped array containing linearized values of the
        upper triangular part (diagonal excluded) of the distance matrix.
        See docs of torch.nn.functional.pdist().
        #TODO #check if pivots set does not contain duplicates, i.e. distances[i,j] != 0  forall i!=j
        """
        distances = np.atleast_2d(distances) 
        #distances = distances.astype(np.double)
        
        n = distances.shape[0]
        base = np.zeros((n, n - 1), dtype=distances.dtype) #dtype=np.double

        base[1, 0] = distances[1, 0]
        for k in trange(2, n, disable=not progress):
            base[k, :k] = self._get_apex(base[:k, :k - 1], distances[k, :k])

        self._base = base

    def embed(self, distances):
        """ Embed objects.

        Args:
            distances (numpy.ndarray): a (n,)- or (B,n)-shaped array containing the distance 
                                      to teh n pivots of the B objects to embed.

        Returns:
            numpy.ndarray: a (B,n)-shaped array containing the embedding. B=1 if input was an 1D-array.
        """
        assert self._base is not None, "Simplex base is not built, call build_base() first"
        return self._get_apex(self._base, distances)

    def estimate(self, x, y, kind='zenit'):
        """ Generate distance estimations from embeddings `x` and `y`.

        Args:
            x (numpy.ndarray): a (n,)- or (B,n)-shaped array of embeddings
            y (numpy.ndarray): a (n,)- or (B,n)-shaped array of embeddings
            kind (str, optional): Kind of estimation. Can be one of 'lower', 'upper', or 'zenit'.  Defaults to 'zenit'.

        Returns:
            numpy.ndarray: a (B,)-shaped array containing distance estimantes.
        """
        assert kind in ('lower', 'upper', 'zenit'), f'Invalid estimate: {kind}'

        x = np.atleast_2d(x)
        y = np.atleast_2d(y)

        # if kind == 'lower':
        #     pass

        if kind == 'upper':
            x = np.concatenate((x[:, :-1], - x[:, -1:]), axis=1)

        elif kind == 'zenit':
            z = np.zeros_like(x[:, :1])
            x = np.concatenate((x[:, :-1], z, x[:, -1:]), axis=1)
            y = np.concatenate((y, z), axis=1)

        return np.sqrt(((x - y) ** 2).sum(axis=1))
