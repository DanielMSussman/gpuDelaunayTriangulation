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

Of note, if you're used to the cellGPU conventions... Compilation is now via cmake rather than the old makefiles.
The current version uses CGAL 5.0.2, which has the virtue of now being a header-only library. This caused a few
major headaches, though, and the use of the flexible "Dscalar" types, which could be switched at compilation between
either floating point or double precision, have been discontinued -- now everything is explicitly either a double or
a float.

More details as this repo shapes up!
