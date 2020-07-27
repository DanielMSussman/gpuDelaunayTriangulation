# gpuDelaunayTriangulation

Lorem ipsum... a repository that provides an implementation of the parallelized, localized Delaunay
triangulation routine described in Chen, Renjie, and Craig Gotsman. "Localizing the delaunay triangulation
and its parallel implementation." Transactions on Computational Science XX. Springer, Berlin, Heidelberg, 2013. 39-55,
focusing on an entirely GPU-centric implementation (see cellGPU by Daniel M. Sussman for a CPU-centric use of the same idea).

Main advance allowing this code was done by Diogo E. P. Pinto; current testing and optimization being done in collaboration with
Daniel M. Sussman.

# Basic compilation

see "INSTALLATION.md" for a few more details.

* $ cd build/
* $ cmake ..
* $ make
