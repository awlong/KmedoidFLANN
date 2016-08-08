# KmedioidFLANN
A K-medoid priority-tree approach for the Fast Library for Approximate Nearest Neighbors (FLANN)

Typical implementations of the FLANN algorithm use either:
1. Kd-Trees: requires a dimensionally additive distance measure
2. K-means: requires constructing a cluster center of mass in some n-dimensional space

These two implementations fail when utilizing more complex distance measures, or in circumstances where a "center of mass" is difficult to define (such as in the field of graph clustering)

Here we utilize the K-medoids algorithm (https://en.wikipedia.org/wiki/K-medoids) to construct a K-medoid priority-tree akin to the K-means priority-tree version of FLANN.

# Features:
- Templated to work on arbitrary data objects with user-defined distance functions
- Sparse distance matrix storage, minimizing repetitive distance computations
- Radius and top M NN search protocols

# Algorithm Sketch: Building a K-medoid tree
1. Take *N* samples *{_x_}<sub>i</sub><sup>N</sup>*, a distance function *f(x<sub>i</sub>,x<sub>j</sub>)*, and branching factor *k*
2. Construct tree root as full ensemble of data points
3. Select *k* samples randomly from _x_ to serve as medoids, _y_
4. Cluster all _x_ points to _y_, C<sub>m</sub>={ x<sub>i</sub> :  *f(x<sub>i</sub>, y<sub>m</sub>)* < *f(x<sub>i</sub>, y<sub>j</sub>)* for all j != m}
5. Find sample with minimum internal distance in C<sub>m</sub>, argmin<sub>C<sub>m,i</sub></sub> Sum<sub>j=1</sub><sup>|C<sub>m</sub>|</sup> *f(C<sub>m,i</sub>, C<sub>m,j</sub>)*
6. Update medoids to the new values found in Step 4.
7. Goto 3 until converged/reached iteration limit
8. For each cluster, C<sub>i</sub>, construct new node in tree as a descendent from the root with only the data points in C<sub>i</sub>
9. Apply this scheme 2-8 recursively, leaf termination occurs when there are less than *k* samples in the cluster.


# TODO:
- Push current branch of code
- Refactor to remove Armadillo dependencies
- Add OpenMP and/or MPI support for parallelized search through the tree
  - Both batch and individual NN search parallelization

# Dependencies:
Built using the Armadillo C++ Linear Algebra Library (http://arma.sourceforge.net/)

# References
[1] M. Muja and D. G. Lowe, "Scalable Nearest Neighbor Algorithms for High Dimensional Data," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 36, no. 11, pp. 2227-2240, Nov. 1 2014.
[2] M. Muja, “Scalable nearest neighbour methods for high dimensional data,” Doctoral Thesis, University of British Columbia, 2013.
