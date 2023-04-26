# nSimplex Projection
Python code to compute the nSimplex projection, that maps metric objects into a finite-dimensional Euclidean space

A set of $n$ reference objects (pivots) $p_1,..,p_n \subset (X,d)$ are used to compute a simplex 'base'. Then for each object $o \in X$ the distances $d(o,p_i)$ are used to project the object into a $n$-dimensional Euclidean space.

## Example (using numpy)
see 'test.py'
```python 
import numpy as np
from scipy.spatial.distance import pdist, cdist, euclidean, squareform

from nsimplex import NSimplex

n_pivots = 256
n_objects = 10
dim = 1024
metric='euclidean'
seed = 7

rng = np.random.default_rng(seed)
pivots = rng.random((n_pivots, dim)) #random pivots
o1 = rng.random((n_objects, dim)) #random objects
o2 = rng.random((n_objects, dim)) #random queries objects

pp = squareform(pdist(pivots, metric=metric)) # pivot-pivot distance matrix with shape (n_pivots, n_pivots)
oo = cdist(o1,o2,metric=metric).diagonal()    # groundtruth distances
o1p = cdist(o1, pivots,metric=metric) # objects-pivots distances with shape (n_objects, n_pivots)
o2p = cdist(o2, pivots,metric=metric) # objects-pivots distances with shape (n_objects, n_pivots)

simplex = NSimplex()  
simplex.build_base(pp) #create base simplex given the pivot-pivot distance matrix 
proj_o1 = simplex.embed(o1p)  # projection of o1, shape (n_objects, n_pivots)
proj_o2 = simplex.embed(o2p)  # projection of o2, shape (n_objects, n_pivots)

# estimate bounds
lwb = simplex.estimate(proj_o1, proj_o2, kind='lower')  # shape (n_objects,)
upb = simplex.estimate(proj_o1, proj_o2, kind='upper')  # shape (n_objects,)
zen = simplex.estimate(proj_o1, proj_o2, kind='zenit')  # shape (n_objects,)
mean = (lwb + upb) / 2
for l, original_dist, m, z, u in zip(lwb, oo, mean, zen, upb):
    print(f'lwb: {l:.5f} <= original_dist: {original_dist:.5f} (mean: {m:.5f} - zen: {z:.5f}) <= upb: {u:.5f}')

```
## Example (using Torch)
```python 
import numpy as np
import torch
from scipy.spatial.distance import pdist, cdist, euclidean, squareform

from nsimplex_torch import NSimplex

n_pivots = 256
n_objects = 10
dim = 1024
seed = 7
device = torch.device('cuda') #'cpu'

torch.manual_seed(seed)
pivots = torch.rand((n_pivots, dim)).to(device)
o1 = torch.rand((n_objects, dim)).to(device)
o2 = torch.rand((n_objects, dim)).to(device)

pp = torch.pdist(pivots)# pivot-pivot distance matrix 
pp = torch.from_numpy(squareform(pp.cpu())).to(device)
        
oo = torch.nn.functional.pairwise_distance(o1, o2) #groundtruth distances

o1p = torch.cdist(o1, pivots) # objects-pivots distances 
o2p = torch.cdist(o2, pivots) # queries-pivots distances 

simplex = NSimplex()  
simplex.build_base(pp, progress=False) #create base simplex given the pivot-pivot distance matrix 
proj_o1 = simplex.embed(o1p)  # projection of o1, shape (n_objects, n_pivots)
proj_o2 = simplex.embed(o2p)  # projection of o2, shape (n_objects, n_pivots)

# estimate bounds
lwb = simplex.estimate(proj_o1, proj_o2, kind='lower')  # shape (n_objects,)
upb = simplex.estimate(proj_o1, proj_o2, kind='upper')  # shape (n_objects,)
zen = simplex.estimate(proj_o1, proj_o2, kind='zenit')  # shape (n_objects,)

mean = (lwb + upb) / 2
for l, original_dist, m, z, u in zip(lwb, oo, mean, zen, upb):
        print(f'lwb: {l:.5f} <= original_dist: {original_dist:.5f} (mean: {m:.5f} - zen: {z:.5f}) <= upb: {u:.5f}')
```


## Math Background
If a metric space $(X,d)$ meets the $(n-1)$-point property then for any $n$  objects sampled from the original space, there exists an $(n-1)$-dimensional     simplex in the Euclidean space whose edge lengths correspond to the  actual distances between the objects.
Let $\sigma=<p_1,..,p_n>$  the simplex generated by the reference objects (pivots)  $p_1,\dots p_n \in (X,d)$  and let $v_1,\dots,v_n$  the corresponding vertices, i.e. $v_1,...,v_n$ are vectors in $R^{n-1}$ s.t. $\ell_2(v_i,v_j)=d(p_i,p_j) \forall i,j=1, \dots, n$

The simpex is  represented by the matrix having $v_i$ as rows: $[v_1;...;v_n]$. For example, the rows of the following matrix represent the coordinates $v_{i,j}$ of four vertices $ v_1,\dots, v_4$ of a tetrahedron in 3D space:

$
\begin{bmatrix}
0		&	0		&	0		\\
v_{2,1}	&	0		&	0		\\
v_{3,1}	&	v_{3,2}	&	0	\\
v_{4,1}	&	v_{4,2}	&	v_{4,3}
\end{bmatrix}
$

The invariant that $v_{i,j} = 0$ whenever $j \ge i$ can be  maintained without loss of generality (for any simplex constructed, this can be achieved by rotation and translation within the Euclidean space while maintaining the distances among all the vertices). Furthermore, if we restrict $v_{i,j} \ge 0$ whenever $j = i-1$ then in each row this component represents the \emph{altitude} of the point $v_i$ with respect to a base simplex formed by $\{ v_1, \dots,  v_{i-1}\}$, which is  represented by the matrix derived by selecting elements above and to the left of the entry $v_{i,j}$.


Given the simplex base, an object $o \in X$, and the distances $d(o,p_i)$  $\forall i=1,\dots,n$, it is possible to project the point $o$ into the vector  $v_o\in R^n$ s.t.  $\ell_2(v_o,v_i)=d(o, p_i)$ for all $i=1,\dots,n$.

For any two objects $q$ and $o$, the projected points $v_o$, $v_q$ satisfies $\ell_2(v_q,v_o)\leq d(o,q)\leq \ell_2(v_q^-,v_o)$ where $v_q^-= \left[v_q[: -1],-v_q[-1]\right]$ (i.e., is the vector $v_q$ in which the last component has the opposite sign)

# Citation
If you use this code in your research, please cite this paper:
```
@article{connor2023nsimplex,
  title={nSimplex Zen: A Novel Dimensionality Reduction for Euclidean and Hilbert Spaces},
  author={Connor, Richard and Vadicamo, Lucia},
  journal={arXiv preprint arXiv:2302.11508},
  year={2023}
}
```
