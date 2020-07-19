#include <cuda_runtime.h>
#include "cellListGPU.cuh"
#include "indexer.h"
#include "periodicBoundaries.h"
#include "functions.h"
#include <iostream>
#include <stdio.h>
#include "DelaunayGPU.cuh"
#include <thrust/sort.h>

/*! \file DelaunayGPU.cu */
/*!
    \addtogroup DelaunayGPUBaseKernels
    @{
*/

#define THREADCOUNT 128


__device__ inline double checkCCW(const double2 pa, const double2 pb, const double2 pc)
    {
    return (pa.x - pb.x) * (pa.y - pc.y) - (pa.y - pb.y) * (pa.x - pc.x);
    }
__device__ inline int checkCW(const double pax, const double pay, const double pbx, const double pby, const double pcx, const double pcy)
    {
    return ((pax - pbx) * (pay - pcy) - (pay - pby) * (pax - pcx)) >0 ? 0 : 1;
    }
__device__ inline unsigned positiveModulo(int i, unsigned n)
    {
    int mod = i % (int) n;
    if(i < 0) mod += n;
    return mod;
    };
/*!
   Is a given cell bucket inside a given edge's angle?
*/
__device__ inline bool cellBucketInsideAngle(const double2 v, const int cx, const int cy, const double2 v1, const double2 v2, const double cellSize, periodicBoundaries &box)
    {
    double2 p1,p2,p3,p4;
    double2 pt = make_double2(0.,0.);
    double2 c1 = make_double2(cx*cellSize,cy*cellSize);
    double2 c2 = make_double2((cx+1)*cellSize,cy*cellSize);
    double2 c3 = make_double2((cx+1)*cellSize,(cy+1)*cellSize);
    double2 c4 = make_double2(cx*cellSize,(cy+1)*cellSize);

    box.minDist(v,c1,p1);
    if(checkCCW(pt, v1, p1)>0 && checkCCW(pt, v2, p1)<0)return true;
    box.minDist(v,c2,p2);
    if(checkCCW(pt, v1, p2)>0 && checkCCW(pt, v2, p2)<0)return true; 
    box.minDist(v,c3,p3);
    if(checkCCW(pt, v1, p3)>0 && checkCCW(pt, v2, p3)<0)return true; 
    box.minDist(v,c4,p4);
    if(checkCCW(pt, v1, p4)>0 && checkCCW(pt, v2, p4)<0)return true; 

    return false;
    }

/*!
  Independently check every triangle in the Delaunay mesh to see if the cirumcircle defined by the
  vertices of that triangle is empty. Use the cell list to ensure that only checks of nearby
  particles are required.
  */
__global__ void gpu_test_circumcenters_kernel(int* __restrict__ d_repair,
                                              const int3* __restrict__ d_circumcircles,
                                              const double2* __restrict__ d_pt,
                                              const unsigned int* __restrict__ d_cell_sizes,
                                              const int* __restrict__ d_cell_idx,
                                              int Nccs,
                                              int xsize,
                                              int ysize,
                                              double boxsize,
                                              periodicBoundaries Box,
                                              Index2D ci,
                                              Index2D cli
                                              )
    {
    // read in the index that belongs to this thread
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= Nccs)
        return;

    //the indices of particles forming the circumcircle
    int3 i1 = d_circumcircles[idx];
    //the vertex we will take to be the origin, and its cell position
    double2 v = d_pt[i1.x];
    int ib=floor(v.x/boxsize);
    int jb=floor(v.y/boxsize);

    double2 pt1,pt2;
    Box.minDist(d_pt[i1.y],v,pt1);
    Box.minDist(d_pt[i1.z],v,pt2);


    //get the circumcircle
    double2 Q;
    double rad;
    Circumcircle(pt1,pt2,Q,rad);

    //look through cells for other particles...re-use pt1 and pt2 variables below
    bool badParticle = false;
    int wcheck = ceil(rad/boxsize)+1;

    if(wcheck > xsize/2) wcheck = xsize/2;
    rad = rad*rad;
    for (int ii = ib-wcheck; ii <= ib+wcheck; ++ii)
        {
        for (int jj = jb-wcheck; jj <= jb+wcheck; ++jj)
            {
            int cx = ii;
            if(cx < 0) cx += xsize;
            if(cx >= xsize) cx -= xsize;
            int cy = jj;
            if(cy < 0) cy += ysize;
            if(cy >= ysize) cy -= ysize;

            int bin = ci(cx,cy);

            for (int pp = 0; pp < d_cell_sizes[bin]; ++pp)
                {
                int newidx = d_cell_idx[cli(pp,bin)];

                Box.minDist(d_pt[newidx],v,pt1);
                Box.minDist(pt1,Q,pt2);

                //if it's in the circumcircle, check that its not one of the three points
                if(pt2.x*pt2.x+pt2.y*pt2.y < rad)
                    {
                    if (newidx != i1.x && newidx != i1.y && newidx !=i1.z)
                        {
                        badParticle = true;
                        d_repair[newidx] = newidx;
                        };
                    };
                };//end loop over particles in the given cell
            };
        };// end loop over cells
    if (badParticle)
        {
          d_repair[i1.x] = i1.x;
          d_repair[i1.y] = i1.y;
          d_repair[i1.z] = i1.z;
        };

    return;
    };

//Gives back the number of points that need to be re-triangulated
__global__ void gpu_size_kernel(
                                    int* __restrict__ d_repair,
                                    int Np,
                                    int* __restrict__ Nf
                                    )
    {
      unsigned int kidx = blockDim.x * blockIdx.x + threadIdx.x;
      if (kidx >= Np)
        return;

      int val1=d_repair[kidx];
      int val2=d_repair[kidx+1];
      if(val1<Np && val2==Np)Nf[0]=kidx+1;
    }


/*!
device function carries out the task of finding a good enclosing polygon, using the virtual point and half-plane intersection method
*/
__device__ void virtual_voronoi_calc_function(        int kidx,
                                              const double2* __restrict__ d_pt,
                                              const unsigned int* __restrict__ d_cell_sizes,
                                              const int* __restrict__ d_cell_idx,
                                              int* __restrict__ P_idx,
                                              double2* __restrict__ P,
                                              double2* __restrict__ Q,
                                              double* __restrict__ Q_rad,
                                              int* __restrict__ d_neighnum,
                                              int Ncells,
                                              int xsize,
                                              int ysize,
                                              double boxsize,
                                              periodicBoundaries &Box,
                                              Index2D &ci,
                                              Index2D &cli,
                                              Index2D &GPU_idx
                                              )
    {
    unsigned int poly_size;
    int m, n;
    double2 pt1;
    double rr;
    double Lmax=(xsize*boxsize)*0.5; 
    double LL=Lmax/1.414213562373095-EPSILON;

    poly_size=4;
    int Hv[4];//={-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
    P[GPU_idx(0, kidx)].x=LL;
    P[GPU_idx(0, kidx)].y=LL;
    P[GPU_idx(1, kidx)].x=-LL;
    P[GPU_idx(1, kidx)].y=LL;
    P[GPU_idx(2, kidx)].x=-LL;
    P[GPU_idx(2, kidx)].y=-LL;
    P[GPU_idx(3, kidx)].x=LL;
    P[GPU_idx(3, kidx)].y=-LL;
    /*
    poly_size=5;
    int Hv[5];//={-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
    P[GPU_idx(0, kidx)].x=0.62*LL;
    P[GPU_idx(0, kidx)].y=1.9*LL;
    P[GPU_idx(1, kidx)].x=-1.62*LL;
    P[GPU_idx(1, kidx)].y=1.176*LL;
    P[GPU_idx(2, kidx)].x=-1.62*LL;
    P[GPU_idx(2, kidx)].y=-1.176*LL;
    P[GPU_idx(3, kidx)].x=.62*LL;
    P[GPU_idx(3, kidx)].y=-1.9*LL;
    P[GPU_idx(4, kidx)].x=2.0*LL;
    P[GPU_idx(4, kidx)].y=0.;
    */
    /*
    poly_size=6;
    int Hv[6];//={-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
    P[GPU_idx(0, kidx)].x=2.*LL;
    P[GPU_idx(0, kidx)].y=0.;
    P[GPU_idx(1, kidx)].x=LL;
    P[GPU_idx(1, kidx)].y=1.7*LL;
    P[GPU_idx(2, kidx)].x=-LL;
    P[GPU_idx(2, kidx)].y=1.7*LL;
    P[GPU_idx(3, kidx)].x=-2.0*LL;
    P[GPU_idx(3, kidx)].y=0.;
    P[GPU_idx(4, kidx)].x=-LL;
    P[GPU_idx(4, kidx)].y=-1.7*LL;
    P[GPU_idx(5, kidx)].x=LL;
    P[GPU_idx(5, kidx)].y=-1.7*LL;
    */

    for(m=0; m<poly_size; m++)
        {
        P_idx[GPU_idx(m, kidx)]=-1;
        n=m+1;
        if(n>=poly_size)n-=poly_size;
        Circumcircle(P[GPU_idx(m,kidx)],P[GPU_idx(n,kidx)], pt1, rr);
        Q[GPU_idx(m,kidx)]=pt1;
        Q_rad[GPU_idx(m,kidx)]=rr;
        }

    double2 disp, pt2, v;
    double xx, yy;
    unsigned int ii, numberInCell, newidx, iii, aa, removed;
    int q, pp, w, j, jj, cx, cy, save_j, cc, dd, cell_rad_in, cell_rad, bin, cell_x, cell_y, save;

#ifdef DEBUGFLAGUP
int blah = 0;
int blah2 = 0;
int blah3=0;
int maxCellsChecked=0;
int spotcheck=18;
int counter= 0 ;
if(kidx==spotcheck) printf("VP initial poly_size = %i\n",poly_size);
#endif

    v = d_pt[kidx];
    bool flag=false;


    for(jj=0; jj<poly_size; jj++)
        {
        pt1=v;//+Q[GPU_idx(jj,kidx)]; //absolute position (within box) of circumcenter
        Box.putInBoxReal(pt1);
        double currentRadius = Q_rad[GPU_idx(jj,kidx)];
        cc = max(0,min(xsize-1,(int)floor(pt1.x/boxsize)));
        dd = max(0,min(ysize-1,(int)floor(pt1.y/boxsize)));
        q = ci(cc,dd);
        //check neighbours of Q's cell inside the circumcircle
        cc = ceil(currentRadius/boxsize);
        cell_rad = min(cc,xsize/2);
        cell_x = q%xsize;
        cell_y = (q - cell_x)/ysize;
#ifdef DEBUGFLAGUP
maxCellsChecked  = max(maxCellsChecked,cell_rad*cell_rad);
counter+=1;
#endif
        for (cell_rad_in = 0; cell_rad_in <= cell_rad; ++cell_rad_in)//check cell buckets in circumcircle
        {
        for (cc = -cell_rad_in; cc <= cell_rad_in; ++cc)//check neigh cc
            {
            for (dd = -cell_rad_in; dd <=cell_rad_in; ++dd)//check neigh dd
                {
                if(cc ==-cell_rad_in ||cc == cell_rad_in ||dd ==-cell_rad_in ||dd==cell_rad_in)
                {

                cx = (cell_x+dd)%xsize;
                if (cx <0)
                    cx+=xsize;
                cy = (cell_y+cc)%ysize;
                if (cy <0)
                    cy+=ysize;

                //check if there are any points in cellsns, if so do change, otherwise go for next bin
                bin = ci(cx,cy);
                numberInCell = d_cell_sizes[bin];

#ifdef DEBUGFLAGUP
//if(kidx==spotcheck) printf("(jj,ff) = (%i,%i)\t counter = %i \t cell_rad_in = %i \t cellIdex = %i\t numberInCell = %i\n", jj,ff,counter,cell_rad_in,bin,numberInCell);
#endif
                for (aa = 0; aa < numberInCell; ++aa)//check parts in cell
                    {
#ifdef DEBUGFLAGUP
blah +=1;
#endif
                    newidx = d_cell_idx[cli(aa,bin)];
                    //6-Compute the half-plane Hv defined by the bissector of v and c, containing c
                    ii=GPU_idx(jj, kidx);
                    iii=GPU_idx((jj+1)%poly_size, kidx);
                    if(newidx==P_idx[ii] || newidx==P_idx[iii] || newidx==kidx)continue;
#ifdef DEBUGFLAGUP
blah2+=1;
#endif
                    //how far is the point from the circumcircle's center?
                    rr=Q_rad[ii]*Q_rad[ii];
                    Box.minDist(d_pt[newidx], v, disp); //disp = vector between new point and the point we're constructing the one ring of
                    Box.minDist(disp,Q[ii],pt1); // pt1 gets overwritten by vector between new point and Pi's circumcenter
                    if(pt1.x*pt1.x+pt1.y*pt1.y>rr)continue;
#ifdef DEBUGFLAGUP
blah3 +=1;
#endif
                    //calculate half-plane bissector
                    if(abs(disp.y)<THRESHOLD)
                        {
                        yy=disp.y/2+1;
                        xx=disp.x/2;
                        }
                    else if(abs(disp.x)<THRESHOLD)
                        {
                        yy=disp.y/2;
                        xx=disp.x/2+1;
                        }
                    else
                        {
                        yy=(disp.y*disp.y+disp.x*disp.x)/(2*disp.y);
                        xx=0;
                        }

                    //7-Q<-Hv intersect Q
                    //8-Update P, based on Q (Algorithm 2)      
                    cx = checkCW(0.5*disp.x,0.5*disp.y,xx,yy,0.,0.);
                    //cy=0; //which side will Q be at
                    j=jj-1;
                    if(j<0)j+=poly_size;
                    m=jj;
                    removed=0;
                    save_j=-1;
                    //see which voronoi temp points fall within the same bisector as cell v
                    for(pp=0; pp<poly_size; pp++)
                        {
                        q=jj-pp;
                        if(q<0)
                            q+=poly_size;
                        cy = checkCW(0.5*disp.x, 0.5*disp.y,xx,yy,Q[GPU_idx(q, kidx)].x,Q[GPU_idx(q, kidx)].y );

                        save=(q+1)%poly_size;
                        if(newidx==P_idx[GPU_idx(q, kidx)] || newidx==P_idx[GPU_idx(save,kidx)])
                            cy=cx+1;

                        Hv[q]=cy;
                        if(cy==cx && save_j==-1)
                            save_j=q;

                        }
                    if(Hv[jj]==cx)
                        continue;

                    //Remove the voronoi test points on the opposite half sector from the cell v
                    //If more than 1 voronoi test point is removed, then also adjust the delaunay neighbors of v
                    for(w=0; w<poly_size; w++)
                        {
                        q=(save_j+w)%poly_size;
                        cy=Hv[q];
                        if(cy!=cx)
                            {
                            switch(removed)
                                {
                                case 0:
                                    j=q;
                                    m=(j+1)%poly_size;
                                    removed++;
                                    break;
                                case 1:
                                    m=(m+1)%poly_size;
                                    removed++;
                                    break;
                                case 2:
                                    for(pp=q; pp<poly_size-1; pp++)
                                        {
                                        Q[GPU_idx(pp,kidx)]=Q[GPU_idx(pp+1,kidx)];
                                        P[GPU_idx(pp,kidx)]=P[GPU_idx(pp+1,kidx)];
                                        Q_rad[GPU_idx(pp,kidx)]=Q_rad[GPU_idx(pp+1,kidx)];
                                        P_idx[GPU_idx(pp,kidx)]=P_idx[GPU_idx(pp+1,kidx)];
                                        Hv[pp]=Hv[pp+1];
                                        }
                                    poly_size--;
                                    if(j>q)j--;
                                    if(save_j>q)save_j--;
                                    m=m%poly_size;
                                    w--;
                                    break;
                                }
                            }
                        else if(removed>0)
                            break;
                        }
                    if(removed==0)
                        continue;

                    //Introduce new (if it exists) delaunay neighbor and new voronoi points
                    Circumcircle(P[GPU_idx(j,kidx)], disp, pt1, xx);
                    Circumcircle(disp, P[GPU_idx(m,kidx)], pt2, yy);
                    if(removed==1)
                        {
                        poly_size++;
                        for(pp=poly_size-2; pp>j; pp--)
                            {
                            Q[GPU_idx(pp+1,kidx)]=Q[GPU_idx(pp,kidx)];
                            P[GPU_idx(pp+1,kidx)]=P[GPU_idx(pp,kidx)];
                            Q_rad[GPU_idx(pp+1,kidx)]=Q_rad[GPU_idx(pp,kidx)];
                            P_idx[GPU_idx(pp+1,kidx)]=P_idx[GPU_idx(pp,kidx)];
                            }
                        }

                    m=(j+1)%poly_size;
                    Q[GPU_idx(m,kidx)]=pt2;
                    Q_rad[GPU_idx(m,kidx)]=yy;
                    P[GPU_idx(m,kidx)]=disp;
                    P_idx[GPU_idx(m,kidx)]=newidx;

                    Q[GPU_idx(j,kidx)]=pt1;
                    Q_rad[GPU_idx(j,kidx)]=xx;
                    flag=true;
                    break;
                    }//end checking all points in the current cell list cell
                if(flag==true)
                    break;
                }//end if spiral check
                }//end cell neighbor check, dd
            if(flag==true)
                break;
            }//end cell neighbor check, cc
        if(flag==true)
            break;   
        }//end cell neighbor check, cell_rad_in
        if(flag==true)
            {
            flag=false;
            }
        }//end iterative loop over all edges of the 1-ring

    d_neighnum[kidx]=poly_size;
#ifdef DEBUGFLAGUP
    if(kidx==spotcheck) printf("VP points checked for kidx %i = %i, ignore self points = %i, ignore points outside circumcircles = %i, total neighs = %i \n",kidx,blah,blah2,blah3,poly_size);
#endif
    }

/*!
device function carries out the task of finding a good enclosing polygon
*/
__device__ void voronoi_calc_function(        int kidx,
                                              const double2* __restrict__ d_pt,
                                              const unsigned int* __restrict__ d_cell_sizes,
                                              const int* __restrict__ d_cell_idx,
                                              int* __restrict__ P_idx,
                                              double2* __restrict__ P,
                                              double2* __restrict__ Q,
                                              double* __restrict__ Q_rad,
                                              int* __restrict__ d_neighnum,
                                              int Ncells,
                                              int xsize,
                                              int ysize,
                                              double boxsize,
                                              periodicBoundaries &Box,
                                              Index2D &ci,
                                              Index2D &cli,
                                              Index2D &GPU_idx
                                              )
    {
    int i,j,k;
    double2 pt1,pt2;
    double rr, z1, z2;
    unsigned int poly_size;
    double Lmax=(xsize*boxsize)/2;

    //construct new poly around i
    double2 v = d_pt[kidx];
    poly_size=4;
    d_neighnum[kidx]=4;
    int Quad[4]={0,0,0,0};
    unsigned int find=0;
    int ii=floor(v.x/boxsize);
    int jj=floor(v.y/boxsize);
    int w=0;
    int bin, cx, cy, quad_angle, numberInCell, newidx, pp, m, n, l, g;
    while(find<4)
        {
        //if(w>Lmax)printf("voro_calc: %d, %d\n", kidx, w);
        //go through the shell
        for (i = -w; i <=w; ++i)
            {
            for (j = -w; j <=w; ++j)
                {
                if(i ==-w ||i == w ||j ==-w ||j==w)
                    {
                    cx = (ii+j)%xsize;
                    if (cx <0)
                        cx+=xsize;
                    cy = (jj+i)%ysize;
                    if (cy <0)
                        cy+=ysize;
                    bin = ci(cx,cy);
                    //go through the cell

                    numberInCell = d_cell_sizes[bin];
                    for (pp = 0; pp < numberInCell; ++pp)
                        {
                        newidx = d_cell_idx[cli(pp,bin)];
                        if(newidx==kidx)continue;

                        Box.minDist(d_pt[newidx],v,pt1);
                        if(pt1.y>0)
                            {
                            if(pt1.x>0)quad_angle=0;
                            else quad_angle=1;
                            }
                        else
                            {
                            if(pt1.x>0)quad_angle=3;
                            else quad_angle=2;
                            }

                        if(Quad[quad_angle]==0)
                            {
                            //check if it is in quadrant
                            P[GPU_idx(quad_angle, kidx)]=pt1;
                            P_idx[GPU_idx(quad_angle, kidx)]=newidx;
                            Quad[quad_angle]=1;
                            find++;

                            //check if it is convex and self-intersecting
                            if(find==4)
                                {
                                for(m=0; m<poly_size; m++)
                                    {
                                    n=m+1;
                                    if(n>=poly_size)n-=poly_size;
                                    Circumcircle(P[GPU_idx(m,kidx)],P[GPU_idx(n,kidx)], pt1, rr);
                                    Q[GPU_idx(m,kidx)]=pt1;
                                    Q_rad[GPU_idx(m,kidx)]=rr;
                                    }

                                    for(m=2; m<=poly_size+1; m++)
                                        {
                                        l=m;
                                        n=m-1;
                                        k=m-2;
                                        if(n>=poly_size)n-=poly_size;
                                        if(l>=poly_size)l-=poly_size;
                                        Box.minDist(Q[GPU_idx(k, kidx)], Q[GPU_idx(n, kidx)],pt1);
                                        Box.minDist(Q[GPU_idx(n, kidx)], Q[GPU_idx(l, kidx)],pt2);
                                        z1=pt1.x*pt2.y-pt1.y*pt2.x;

                                        //check if it is convex
                                        if(m>2 && z1/z2<0)
                                            {
                                            find-=1;
                                            Quad[n]=0;
                                            break;
                                            }
                                        z2=z1;
                                        g=m+1;
                                        if(g>=poly_size)
                                            g-=poly_size;

                                        //check if it is self-intersecting
                                        if(ccw(Q[GPU_idx(k, kidx)],Q[GPU_idx(l, kidx)],Q[GPU_idx(g, kidx)])!=ccw(Q[GPU_idx(n, kidx)],Q[GPU_idx(l, kidx)],Q[GPU_idx(g, kidx)]) && ccw(Q[GPU_idx(k, kidx)],Q[GPU_idx(n, kidx)],Q[GPU_idx(l, kidx)])!=ccw(Q[GPU_idx(k, kidx)],Q[GPU_idx(n, kidx)],Q[GPU_idx(g, kidx)]))
                                            {
                                            find-=1;
                                            Quad[n]=0;
                                            break;
                                            }
                                        }
                                }//end if find
                            }//end if Quad
                        if(find==4)break;
                        }//end pt check
                    }//end in if ij 
                if(find==4)break;
                }//end j
            if(find==4)break;
            }//end i
        w++;

        //if the routine was not able to find a valid polygon around the cell 
        //then create 4 virtual points at "infinity" that form a quadrilateral
        if(w>=Lmax)
            {
            double LL=Lmax/1.414213562373095-EPSILON;
            P[GPU_idx(0, kidx)].x=LL;
            P[GPU_idx(0, kidx)].y=LL;
            P[GPU_idx(1, kidx)].x=-LL;
            P[GPU_idx(1, kidx)].y=LL;
            P[GPU_idx(2, kidx)].x=-LL;
            P[GPU_idx(2, kidx)].y=-LL;
            P[GPU_idx(3, kidx)].x=LL;
            P[GPU_idx(3, kidx)].y=-LL;
            for(m=0; m<poly_size; m++)
                {
                P_idx[GPU_idx(m, kidx)]=-1;
                n=m+1;
                if(n>=poly_size)n-=poly_size;
                Circumcircle(P[GPU_idx(m,kidx)],P[GPU_idx(n,kidx)], pt1, rr);
                Q[GPU_idx(m,kidx)]=pt1;
                Q_rad[GPU_idx(m,kidx)]=rr;
                }
            find=4;
            }

        }//end while

    }

//assumes "fixlist" has the structure fixlist[ii]=-1 --> dont triangulate
__global__ void gpu_voronoi_calc_no_sort_kernel(const double2* __restrict__ d_pt,
                                              const unsigned int* __restrict__ d_cell_sizes,
                                              const int* __restrict__ d_cell_idx,
                                              int* __restrict__ P_idx,
                                              double2* __restrict__ P,
                                              double2* __restrict__ Q,
                                              double* __restrict__ Q_rad,
                                              int* __restrict__ d_neighnum,
                                              int Ncells,
                                              int xsize,
                                              int ysize,
                                              double boxsize,
                                              periodicBoundaries Box,
                                              Index2D ci,
                                              Index2D cli,
                                              const int* __restrict__ d_fixlist,
                                              Index2D GPU_idx
                                              )
    {
    unsigned int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx >= Ncells)return;
    if(d_fixlist[tidx] >= 0)
        virtual_voronoi_calc_function(tidx,d_pt,d_cell_sizes,d_cell_idx,
                          P_idx, P, Q, Q_rad,
                          d_neighnum,
                          Ncells, xsize,ysize, boxsize,Box,
                          ci,cli,GPU_idx);
    return;
    }
/*
GPU implementatio of the DT. It makes use of a locallity lema described in (doi: 10.1109/ISVD.2012.9). It will only make the repair of the topology in case it is necessary. Steps are detailed as in paper.
*/
//This kernel constructs the initial test polygon.
//Currently it only uses 4 points, one in each quadrant.
//The initial test voronoi cell needs to be valid for the algorithm to work.
//Thus if the search fails, 4 virtual points are used at maximum distance as the starting polygon
__global__ void gpu_voronoi_calc_kernel(const double2* __restrict__ d_pt,
                                              const unsigned int* __restrict__ d_cell_sizes,
                                              const int* __restrict__ d_cell_idx,
                                              int* __restrict__ P_idx,
                                              double2* __restrict__ P,
                                              double2* __restrict__ Q,
                                              double* __restrict__ Q_rad,
                                              int* __restrict__ d_neighnum,
                                              int Ncells,
                                              int xsize,
                                              int ysize,
                                              double boxsize,
                                              periodicBoundaries Box,
                                              Index2D ci,
                                              Index2D cli,
                                              const int* __restrict__ d_fixlist,
                                              int Nf,
                                              Index2D GPU_idx
                                              )
    {
    unsigned int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx >= Nf)return;
    unsigned int kidx=d_fixlist[tidx];

   virtual_voronoi_calc_function(kidx,d_pt,d_cell_sizes,d_cell_idx,
                          P_idx, P, Q, Q_rad,
                          d_neighnum,
                          Ncells, xsize,ysize, boxsize,Box,
                          ci,cli,GPU_idx);
    return;
    }

//!Global voro calc kernel does not need a fixlist
__global__ void gpu_voronoi_calc_global_kernel(const double2* __restrict__ d_pt,
                                              const unsigned int* __restrict__ d_cell_sizes,
                                              const int* __restrict__ d_cell_idx,
                                              int* __restrict__ P_idx,
                                              double2* __restrict__ P,
                                              double2* __restrict__ Q,
                                              double* __restrict__ Q_rad,
                                              int* __restrict__ d_neighnum,
                                              int Ncells,
                                              int xsize,
                                              int ysize,
                                              double boxsize,
                                              periodicBoundaries Box,
                                              Index2D ci,
                                              Index2D cli,
                                              Index2D GPU_idx
                                              )
    {
    unsigned int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx >= Ncells)return;

    virtual_voronoi_calc_function(tidx,d_pt,d_cell_sizes,d_cell_idx,
                          P_idx, P, Q, Q_rad,
                          d_neighnum,
                          Ncells, xsize,ysize, boxsize,Box,
                          ci,cli,GPU_idx);
    return;
    }

/*!
device function that goes from a candidate 1-ring to an actual 1-ring
*/
__device__ void get_oneRing_function_alternate(int kidx,
                const double2* __restrict__ d_pt,
                const unsigned int* __restrict__ d_cell_sizes,
                const int* __restrict__ d_cell_idx,
                int* __restrict__ P_idx,
                double2* __restrict__ P,
                double2* __restrict__ Q,
                double* __restrict__ Q_rad,
                int* __restrict__ d_neighnum,
                int Ncells,
                int xsize,
                int ysize,
                double boxsize,
                periodicBoundaries &Box,
                Index2D &ci,
                Index2D &cli,
                Index2D &GPU_idx,
                int const currentMaxNeighbors,
                int *maximumNeighborNumber
                )
    {

    //I will reuse most variables
    double2 disp, pt1, pt2, v,currentQ;// v1, v2;
    double rr, xx, yy,currentRadius;
    unsigned int numberInCell, newidx, aa, removed;
    int q, pp, m, w, j, jj, cx, cy, cc, dd, cell_rad_in, cell_rad, bin, cell_x, cell_y;
    unsigned int poly_size=d_neighnum[kidx];

    v = d_pt[kidx];
    bool flag=false;

    __shared__ double2 sd2[THREADCOUNT];
    __shared__ double sd[THREADCOUNT];
    __shared__ int si[THREADCOUNT];

    int baseIdx = GPU_idx(0,kidx);
    for(jj=0; jj<poly_size; jj++)
        {
        currentQ = Q[baseIdx+jj];
        currentRadius = Q_rad[baseIdx+jj];
        pt1=v+currentQ; //absolute position (within box) of circumcenter
        //v1=P[GPU_idx(jj, kidx)];
        //v2=P[GPU_idx((jj+1)%poly_size, kidx)];
        Box.putInBoxReal(pt1);
        cc = max(0,min(xsize-1,(int)floor(pt1.x/boxsize)));
        dd = max(0,min(ysize-1,(int)floor(pt1.y/boxsize)));
        q = ci(cc,dd);
        //check neighbours of Q's cell inside the circumcircle
        cc = ceil(currentRadius/boxsize);
        cell_rad = min(cc,xsize/2);
        cell_x = q%xsize;
        cell_y = (q - cell_x)/ysize;
        for (cell_rad_in = 0; cell_rad_in <= cell_rad; ++cell_rad_in)//check cell buckets in circumcircle
        {
        for (cc = -cell_rad_in; cc <= cell_rad_in; ++cc)//check neigh cc
            {
            for (dd = -cell_rad_in; dd <=cell_rad_in; ++dd)//check neigh dd
                {
                if(cc ==-cell_rad_in ||cc == cell_rad_in ||dd ==-cell_rad_in ||dd==cell_rad_in)
                {

                cx = positiveModulo(cell_x+dd,xsize); 
                cy = positiveModulo(cell_y+cc,ysize); 

                //if(cellBucketInsideAngle(v, cx, cy, v1, v2, boxsize, Box)==false)continue;

                //check if there are any points in cellsns, if so do change, otherwise go for next bin
                bin = ci(cx,cy);
                numberInCell = d_cell_sizes[bin];

                //if(kidx==spotcheck) printf("(jj,ff) = (%i,%i)\t counter = %i \t cell_rad_in = %i \t cellIdex = %i\t numberInCell = %i\n",
                //                            jj,ff,counter,cell_rad_in,bin,numberInCell);

                for (aa = 0; aa < numberInCell; ++aa)//check parts in cell
                    {
                    newidx = d_cell_idx[cli(aa,bin)];
                    if(newidx == kidx) continue;
                    bool skipPoint = false;
                    for (int pidx = 0; pidx < poly_size; ++pidx)
                        if(newidx == P_idx[baseIdx+pidx]) skipPoint = true;
                    if (skipPoint) continue;
                    //6-Compute the half-plane Hv defined by the bissector of v and c, containing c
                    //how far is the point from the circumcircle's center?
                    rr=currentRadius*currentRadius;
                    Box.minDist(d_pt[newidx], v, disp); //disp = vector between new point and the point we're constructing the one ring of
                    Box.minDist(disp,currentQ,pt1); // pt1 gets overwritten by vector between new point and Pi's circumcenter
                    if(pt1.x*pt1.x+pt1.y*pt1.y>rr)continue;
                    //calculate half-plane bissector
                    if(abs(disp.y)<THRESHOLD)
                        {
                        yy=disp.y/2+1;
                        xx=disp.x/2;
                        }
                    else if(abs(disp.x)<THRESHOLD)
                        {
                        yy=disp.y/2;
                        xx=disp.x/2+1;
                        }
                    else
                        {
                        yy=(disp.y*disp.y+disp.x*disp.x)/(2*disp.y);
                        xx=0;
                        }

                    //7-Q<-Hv intersect Q
                    //8-Update P, based on Q (Algorithm 2)      
                    cx = checkCW(0.5*disp.x,0.5*disp.y,xx,yy,0.,0.);
                    if(cx== checkCW(0.5*disp.x, 0.5*disp.y,xx,yy,Q[baseIdx+jj].x,Q[baseIdx+jj].y))
                        continue;

                    //Remove the voronoi test points on the opposite half sector from the cell v
                    //If more than 1 voronoi test point is removed, then also adjust the delaunay neighbors of v
                    int removeCW=0;
                    bool removeCCW=false;
                    bool firstRemove=true;
                    removed=0;
                    j=-1;
                    //which side will Q be at
                    cy = checkCW(0.5*disp.x, 0.5*disp.y,xx,yy,Q[baseIdx+poly_size-1].x,Q[baseIdx+poly_size-1].y);

                    removeCW=cy;
                    if(cy!=cx)
                        {
                        j=poly_size-1;
                        removed++;
                        removeCCW=true;
                        }

                    for(w=0; w<poly_size-1; w++)
                        {
                        cy = checkCW(0.5*disp.x, 0.5*disp.y,xx,yy,Q[baseIdx+w].x,Q[baseIdx+w].y);
                        if(cy!=cx)
                            {
                            if(removeCCW==false)
                                {
                                if(j<0)
                                    j=w;
                                else if(j>w)
                                    j=w;
                                removed++;
                                removeCCW=true;
                                }
                            else
                                {
                                if(firstRemove==false)
                                    {
                                    //if(kidx ==127 ) printf("FR kidx %i shifting poly by %i\n",kidx,poly_size-1-w);
                                    for(pp=w; pp<poly_size-1; pp++)
                                        {
                                        sd2[threadIdx.x] = Q[baseIdx+pp+1];
                                        Q[baseIdx+pp] = sd2[threadIdx.x];
                                        //Q[baseIdx+pp]=Q[baseIdx+pp+1];
                                        }
                                    for(pp=w; pp<poly_size-1; pp++)
                                        {
                                        sd2[threadIdx.x] = P[baseIdx+pp+1];
                                        P[baseIdx+pp ] = sd2[threadIdx.x];
                                        //P[baseIdx+pp]=P[baseIdx+pp+1];
                                        }
                                    for(pp=w; pp<poly_size-1; pp++)
                                        {
                                        sd[threadIdx.x] = Q_rad[baseIdx+pp+1];
                                        Q_rad[baseIdx+pp] = sd[threadIdx.x];
                                        //Q_rad[baseIdx+pp]=Q_rad[baseIdx+pp+1];
                                        }
                                    for(pp=w; pp<poly_size-1; pp++)
                                        {
                                        si[threadIdx.x] = P_idx[baseIdx+pp+1];
                                        P_idx[baseIdx+pp ] = si[threadIdx.x];
                                        //P_idx[baseIdx+pp]=P_idx[baseIdx+pp+1];
                                        }
                                    poly_size--;
                                    if(j>w)
                                        j--;
                                    w--;
                                    }
                                else 
                                    firstRemove=false;
                                removed++;
                                }	    
                            }
                        else
                            removeCCW=false;
                        }
                    if(removeCW!=cx && removeCCW==true && firstRemove==false)
                        {
                        poly_size--;
                        if(j>w)j--;
                        }

                    if(removed==0)
                        continue;

                    //Introduce new (if it exists) delaunay neighbor and new voronoi points
                    if(removed>1)m=(j+2)%poly_size;
                    else m=(j+1)%poly_size;
                    Circumcircle(P[baseIdx+j], disp, pt1, xx);
                    Circumcircle(disp, P[baseIdx+m], pt2, yy);
                    if(removed==1)
                        {
                        //if(kidx ==18 ) printf("kidx %i shifting poly by %i\n",kidx,poly_size-1-j);
                        poly_size++;
                        for(pp=poly_size-2; pp>j; pp--)
                            {
                            sd2[threadIdx.x] = Q[baseIdx+pp];
                            Q[baseIdx+pp+1] = sd2[threadIdx.x];
                        //    testDouble2 = Q[baseIdx+pp];
                          //  Q[baseIdx+pp+1]=testDouble2;
                            //Q[baseIdx+pp];
                            }
                        for(pp=poly_size-2; pp>j; pp--)
                            {
                            si[threadIdx.x] = P_idx[baseIdx+pp];
                            P_idx[baseIdx+pp+1] = si[threadIdx.x];
                            //testInt = P_idx[baseIdx+pp];
                            //P_idx[baseIdx+pp+1]=testInt;
                            //P_idx[baseIdx+pp];
                            }
                        for(pp=poly_size-2; pp>j; pp--)
                            {
                            sd2[threadIdx.x] = P[baseIdx+pp];
                            P[baseIdx+pp+1] = sd2[threadIdx.x];
                            //testDouble2 = P[baseIdx+pp];
                            //P[baseIdx+pp+1]=testDouble2;
                            //P[baseIdx+pp];
                            }
                        for(pp=poly_size-2; pp>j; pp--)
                            {
                            sd[threadIdx.x] = Q_rad[baseIdx+pp];
                            Q_rad[baseIdx+pp+1] = sd[threadIdx.x];
                            //testDouble = Q_rad[baseIdx+pp];
                            //Q_rad[baseIdx+pp+1]=testDouble;
                            //Q_rad[baseIdx+pp];
                            }
                        }
                    m=(j+1)%poly_size;
                    Q[baseIdx+m]=pt2;
                    Q_rad[baseIdx+m]=yy;
                    P[baseIdx+m]=disp;
                    P_idx[baseIdx+m]=newidx;

                    Q[baseIdx+j]=pt1;
                    Q_rad[baseIdx+j]=xx;
                    flag=true;
                    break;
                    }//end checking all points in the current cell list cell
                if(flag==true)
                    break;
                }//end if spiral check
                }//end cell neighbor check, dd
            if(flag==true)
                    break;
            }//end cell neighbor check, cc
        if(flag==true)
                break;   
        }//end cell neighbor check, cell_rad_in
        if(flag==true)
        {
                jj--;
                flag=false;
        }
        }//end iterative loop over all edges of the 1-ring

    d_neighnum[kidx]=poly_size;

    return;
    }//end function

/*!
  device function that goes from a candidate 1-ring to an actual 1-ring
 */
__device__ void get_oneRing_function(int kidx,
                const double2* __restrict__ d_pt,
                const unsigned int* __restrict__ d_cell_sizes,
                const int* __restrict__ d_cell_idx,
                int* __restrict__ P_idx,
                double2* __restrict__ P,
                double2* __restrict__ Q,
                double* __restrict__ Q_rad,
                int* __restrict__ d_neighnum,
                int Ncells,
                int xsize,
                int ysize,
                double boxsize,
                periodicBoundaries &Box,
                Index2D &ci,
                Index2D &cli,
                Index2D &GPU_idx,
                int const currentMaxNeighbors,
                int *maximumNeighborNumber
                )
{
        double2 disp, pt1, pt2, v,currentQ;
        //    double2 v1,v2;
        double rr, xx, yy,currentRadius;
        unsigned int numberInCell, newidx, aa, removed;
        int q, pp, m, w, j, jj, cx, cy, save_j, cc, dd, cell_rad_in, bin, cell_x, cell_y;
        unsigned int poly_size=d_neighnum[kidx];

        v = d_pt[kidx];
        bool flag=false;

#ifdef DEBUGFLAGUP
        int blah = 0;
        int blah2 = 0;
        int blah3=0;
        int maxCellsChecked=0;
        int spotcheck=18;
        int counter= 0 ;
        if(kidx==spotcheck) printf("VP initial poly_size = %i\n",poly_size);
#endif

        int baseIdx = GPU_idx(0,kidx);

        for(jj=0; jj<poly_size; jj++)
        {
                currentQ = Q[baseIdx+jj];
                currentRadius = Q_rad[baseIdx+jj];
                pt1=v+currentQ; //absolute position (within box) of circumcenter
                //        v1=P[GPU_idx(jj, kidx)];
                //        v2=P[GPU_idx((jj+1)%poly_size,kidx)];
                Box.putInBoxReal(pt1);

                cc = max(0,min(xsize-1,(int)floor(pt1.x/boxsize)));
                dd = max(0,min(ysize-1,(int)floor(pt1.y/boxsize)));
                q = ci(cc,dd);
                //check neighbours of Q's cell inside the circumcircle
                cc = ceil(currentRadius/boxsize);
                cell_rad_in = min(cc,xsize/2);
                cell_x = q%xsize;
                cell_y = (q - cell_x)/ysize;
#ifdef DEBUGFLAGUP
                counter+=1;
                maxCellsChecked  = max(maxCellsChecked,cell_rad_in*cell_rad_in);
#endif

                //This ordering of how the cells are checked is clearly inefficient -- perhaps replace it by a biased
                //ordereding depending on where the circumcenter is relative to the target point
                for (int cri = 0; cri <= cell_rad_in; ++cri)
                {
                        for (cc = -cri; cc <= cri; ++cc)//check neigh i
                        {
                                for (dd = -cri; dd <=cri; ++dd)//check neigh q
                                {
                                        if(!(cc==-cri || cc == cri || dd == -cri || dd == cri))
                                                continue;
                                        cx = positiveModulo(cell_x+dd,xsize); 
                                        cy = positiveModulo(cell_y+cc,ysize); 
                                        //if(!cellBucketInsideAngle(v,cx,cy,v1,v2,boxsize,Box))continue;
                                        //check if there are any points in cellsns, if so do change, otherwise go for next bin
                                        bin = ci(cx,cy);
                                        numberInCell = d_cell_sizes[bin];

#ifdef DEBUGFLAGUP
                                        //if(kidx==spotcheck) printf("(jj,ff) = (%i,%i)\t counter = %i \t cell_rad_in = %i \t cellIdex = %i\t numberInCell = %i\n",jj,ff,counter,cell_rad_in,bin,numberInCell);
#endif

                                        for (aa = 0; aa < numberInCell; ++aa)//check parts in cell
                                        {
#ifdef DEBUGFLAGUP
                                                blah +=1;
#endif
                                                newidx = d_cell_idx[cli(aa,bin)];
                                                //skip any newidx's that are already part of the 1-ring
                                                if(newidx == kidx) continue;
                                                bool skipPoint = false;
                                                for (int pidx = 0; pidx < poly_size; ++pidx)
                                                        if(newidx == P_idx[baseIdx+pidx]) skipPoint = true;
                                                if (skipPoint) continue;

                                                //6-Compute the half-plane Hv defined by the bissector of v and c, containing c
#ifdef DEBUGFLAGUP
                                                blah2+=1;
#endif
                                                //how far is the point from the circumcircle's center?
                                                rr=currentRadius*currentRadius;
                                                Box.minDist(d_pt[newidx], v, disp); //disp = vector between new point and the point we're constructing the one ring of
                                                Box.minDist(disp,currentQ,pt1); // pt1 gets overwritten by vector between new point and Pi's circumcenter
                                                if(pt1.x*pt1.x+pt1.y*pt1.y>rr) continue;
#ifdef DEBUGFLAGUP
                                                blah3 +=1;
#endif
                                                //calculate half-plane bissector
                                                if(abs(disp.y)<THRESHOLD)
                                                {
                                                        yy=disp.y/2+1;
                                                        xx=disp.x/2;
                        }
                    else if(abs(disp.x)<THRESHOLD)
                        {
                        yy=disp.y/2;
                        xx=disp.x/2+1;
                        }
                    else
                        {
                        yy=(disp.y*disp.y+disp.x*disp.x)/(2*disp.y);
                        xx=0;
                        }

                    //7-Q<-Hv intersect Q
                    //8-Update P, based on Q (Algorithm 2)      
                    //which side is v at?
                    cx = checkCW(0.5*disp.x,0.5*disp.y,xx,yy,0.,0.);
                    if(cx== checkCW(0.5*disp.x, 0.5*disp.y,xx,yy,Q[baseIdx+jj].x,Q[baseIdx+jj].y))
                        continue;

                    //which side will Q be at
                    j= jj > 0 ? jj-1 : poly_size - 1;
                    m=jj;
                    removed=0;
                    save_j=-1;
                    //see which voronoi temp points fall within the same bisector as cell v
                    for(q = poly_size-1;q >=0; q--)
                        {
                        if(save_j == -1 && cx==checkCW(0.5*disp.x, 0.5*disp.y,xx,yy,Q[baseIdx+q].x,Q[baseIdx+q].y ))
                            {
                            save_j=q;
                            break;
                            }
                        }

                    //Remove the voronoi test points on the opposite half sector from the cell v
                    //If more than 1 voronoi test point is removed, then also adjust the delaunay neighbors of v
                    for(w=0; w<poly_size; w++)
                        {
                        q=(save_j+w)%poly_size;
                        cy = checkCW(0.5*disp.x, 0.5*disp.y,xx,yy,Q[baseIdx+q].x,Q[baseIdx+q].y);
                        if(cy!=cx)
                            {
                            switch(removed)
                                {
                                case 0:
                                    j=q;
                                    m=(j+1)%poly_size;
                                    removed++;
                                    break;
                                case 1:
                                    m=(m+1)%poly_size;
                                    removed++;
                                    break;
                                case 2:
                                    for(pp=q; pp<poly_size-1; pp++)
                                        {
                                        Q[baseIdx+pp] = Q[baseIdx+pp+1];
                                        P[baseIdx+pp] = P[baseIdx+pp+1];
                                        Q_rad[baseIdx+pp] = Q_rad[baseIdx+pp+1];
                                        P_idx[baseIdx+pp] = P_idx[baseIdx+pp+1];
                                        }
                                    poly_size--;
                                    if(j>q)j--;
                                    if(save_j>q)save_j--;
                                    m=m%poly_size;
                                    w--;
                                    break;
                                }
                            }
                        else if(removed>0)
                            break;
                        }
                    if(removed==0)
                        continue;

                    //Introduce new (if it exists) delaunay neighbor and new voronoi points
                    Circumcircle(P[GPU_idx(j,kidx)], disp, pt1, xx);
                    Circumcircle(disp, P[GPU_idx(m,kidx)], pt2, yy);
                    if(removed==1)
                        {
                        poly_size++;
                        if(poly_size > currentMaxNeighbors)
                            {
                            atomicMax(&maximumNeighborNumber[0],poly_size);
                            //printf("excess (%i,%i,%i)\n",poly_size,currentMaxNeighbors,maximumNeighborNumber[0]);
                            return;
                            }
                        for(pp=poly_size-2; pp>j; pp--)
                            {
                            Q[baseIdx+pp+1]=Q[baseIdx+pp];
                            P[baseIdx+pp+1]=P[baseIdx+pp];
                            Q_rad[baseIdx+pp+1]=Q_rad[baseIdx+pp];
                            P_idx[baseIdx+pp+1]=P_idx[baseIdx+pp];
                            }
                        }

                    m=(j+1)%poly_size;
                    Q[GPU_idx(m,kidx)]=pt2;
                    Q_rad[GPU_idx(m,kidx)]=yy;
                    P[GPU_idx(m,kidx)]=disp;
                    P_idx[GPU_idx(m,kidx)]=newidx;

                    Q[GPU_idx(j,kidx)]=pt1;
                    Q_rad[GPU_idx(j,kidx)]=xx;
                    flag=true;
                    break;
                    }//end checking all points in the current cell list cell
                if(flag==true)
                    break;
                }//end cell neighbor check, q
            if(flag==true)
                break;
            }//end cell neighbor check, i
        }
        if(flag==true)
            {
            jj--;
            flag=false;
            }
        }//end iterative loop over all edges of the 1-ring

    d_neighnum[kidx]=poly_size;
#ifdef DEBUGFLAGUP
if(kidx==spotcheck) printf(" points checked for kidx %i = %i, ignore self points = %i, ignore points outside circumcircles = %i, total neighs = %i \n",kidx,blah,blah2,blah3,poly_size);
#endif
    }

//This kernel updates the initial polygon into the real delaunay one.
//It goes through the same steps as in the paper, using the half plane intersection routine.
//It outputs the complete triangulation per point in CCW order
__global__ void gpu_get_neighbors_kernel(const double2* __restrict__ d_pt,
                const unsigned int* __restrict__ d_cell_sizes,
                const int* __restrict__ d_cell_idx,
                int* __restrict__ P_idx,
                double2* __restrict__ P,
                double2* __restrict__ Q,
                double* __restrict__ Q_rad,
                int* __restrict__ d_neighnum,
                int Ncells,
                int xsize,
                int ysize,
                double boxsize,
                periodicBoundaries Box,
                Index2D ci,
                Index2D cli,
                const int* __restrict__ d_fixlist,
                int Nf,
                Index2D GPU_idx,
                int *maximumNeighborNum,
                int currentMaxNeighborNum
                )
    {


    unsigned int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx >= Nf)return;
    unsigned int kidx=d_fixlist[tidx];
    if (kidx >= Ncells)return;

    get_oneRing_function_alternate(kidx, d_pt,d_cell_sizes,d_cell_idx,P_idx, P,Q,Q_rad,d_neighnum, Ncells,xsize,ysize,boxsize,Box,ci,cli,GPU_idx, currentMaxNeighborNum,maximumNeighborNum);
    //get_oneRing_function(kidx, d_pt,d_cell_sizes,d_cell_idx,P_idx, P,Q,Q_rad,d_neighnum, Ncells,xsize,ysize,boxsize,Box,ci,cli,GPU_idx, currentMaxNeighborNum,maximumNeighborNum);

    return;
    }//end function

__global__ void gpu_get_neighbors_no_sort_kernel(const double2* __restrict__ d_pt,
                const unsigned int* __restrict__ d_cell_sizes,
                const int* __restrict__ d_cell_idx,
                int* __restrict__ P_idx,
                double2* __restrict__ P,
                double2* __restrict__ Q,
                double* __restrict__ Q_rad,
                int* __restrict__ d_neighnum,
                int Ncells,
                int xsize,
                int ysize,
                double boxsize,
                periodicBoundaries Box,
                Index2D ci,
                Index2D cli,
                const int* __restrict__ d_fixlist,
                Index2D GPU_idx,
                int *maximumNeighborNum,
                int currentMaxNeighborNum
                )
    {


    unsigned int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx >= Ncells)return;
    if(d_fixlist[tidx] >=0)
        get_oneRing_function_alternate(tidx, d_pt,d_cell_sizes,d_cell_idx,P_idx, P,Q,Q_rad,d_neighnum, Ncells,xsize,ysize,boxsize,Box,ci,cli,GPU_idx, currentMaxNeighborNum,maximumNeighborNum);

    return;
    }//end function

//!global get neighbors does not need a fixlist
__global__ void gpu_get_neighbors_global_kernel(const double2* __restrict__ d_pt,
                const unsigned int* __restrict__ d_cell_sizes,
                const int* __restrict__ d_cell_idx,
                int* __restrict__ P_idx,
                double2* __restrict__ P,
                double2* __restrict__ Q,
                double* __restrict__ Q_rad,
                int* __restrict__ d_neighnum,
                int Ncells,
                int xsize,
                int ysize,
                double boxsize,
                periodicBoundaries Box,
                Index2D ci,
                Index2D cli,
                Index2D GPU_idx,
                int *maximumNeighborNum,
                int currentMaxNeighborNum
                )
    {


    unsigned int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx >= Ncells)return;

    get_oneRing_function_alternate(tidx, d_pt,d_cell_sizes,d_cell_idx,P_idx, P,Q,Q_rad,d_neighnum, Ncells,xsize,ysize,boxsize,Box,ci,cli,GPU_idx, currentMaxNeighborNum,maximumNeighborNum);
    //get_oneRing_function(tidx, d_pt,d_cell_sizes,d_cell_idx,P_idx, P,Q,Q_rad,d_neighnum, Ncells,xsize,ysize,boxsize,Box,ci,cli,GPU_idx, currentMaxNeighborNum,maximumNeighborNum);
        
    return;
    }//end function



/////////////////////////////////////////////////////////////
//////
//////			Kernel Calls
//////
/////////////////////////////////////////////////////////////

bool gpu_voronoi_calc_no_sort(double2* d_pt,
                      unsigned int* d_cell_sizes,
                      int* d_cell_idx,
                      int* P_idx,
                      double2* P,
                      double2* Q,
                      double* Q_rad,
                      int* d_neighnum,
                      int Ncells,
                      int xsize,
                      int ysize,
                      double boxsize,
                      periodicBoundaries Box,
                      Index2D ci,
                      Index2D cli,
                      int* d_fixlist,
                      Index2D GPU_idx
                      )
    {
    unsigned int block_size = THREADCOUNT;
    if (Ncells < THREADCOUNT) block_size = 32;
    unsigned int nblocks  = Ncells/block_size + 1;
    gpu_voronoi_calc_no_sort_kernel<<<nblocks,block_size>>>(
                        d_pt,
                        d_cell_sizes,
                        d_cell_idx,
                        P_idx,
                        P,
                        Q,
                        Q_rad,
                        d_neighnum,
                        Ncells,
                        xsize,
                        ysize,
                        boxsize,
                        Box,
                        ci,
                        cli,
                        d_fixlist,
                        GPU_idx
                        );
    HANDLE_ERROR(cudaGetLastError());
#ifdef DEBUGFLAGUP
    cudaDeviceSynchronize();
#endif
    return cudaSuccess;
    }

bool gpu_voronoi_calc(double2* d_pt,
                unsigned int* d_cell_sizes,
                int* d_cell_idx,
                int* P_idx,
                double2* P,
                double2* Q,
                double* Q_rad,
                int* d_neighnum,
                int Ncells,
                int xsize,
                int ysize,
                double boxsize,
                periodicBoundaries Box,
                Index2D ci,
                Index2D cli,
                int* d_fixlist,
                int Nf,
                Index2D GPU_idx,
                bool globalRoutine
                )
{
        unsigned int block_size = THREADCOUNT;
        if (Nf < THREADCOUNT) block_size = 32;
        unsigned int nblocks  = Nf/block_size + 1;

        if(globalRoutine)
            gpu_voronoi_calc_global_kernel<<<nblocks,block_size>>>(
                        d_pt,
                        d_cell_sizes,
                        d_cell_idx,
                        P_idx,
                        P,
                        Q,
                        Q_rad,
                        d_neighnum,
                        Ncells,
                        xsize,
                        ysize,
                        boxsize,
                        Box,
                        ci,
                        cli,
                        GPU_idx
                        );
        else
            gpu_voronoi_calc_kernel<<<nblocks,block_size>>>(
                        d_pt,
                        d_cell_sizes,
                        d_cell_idx,
                        P_idx,
                        P,
                        Q,
                        Q_rad,
                        d_neighnum,
                        Ncells,
                        xsize,
                        ysize,
                        boxsize,
                        Box,
                        ci,
                        cli,
                        d_fixlist,
                        Nf,
                        GPU_idx
                        );

        HANDLE_ERROR(cudaGetLastError());
#ifdef DEBUGFLAGUP

    cudaDeviceSynchronize();
#endif
        return cudaSuccess;
};

bool gpu_get_neighbors_no_sort(double2* d_pt, //the point set
                unsigned int* d_cell_sizes,//points per bucket
                int* d_cell_idx,//cellListIdxs
                int* P_idx,//index of Del Neighbors
                double2* P,//location del neighborPositions
                double2* Q,//voronoi vertex positions
                double* Q_rad,//radius? associated with voro vertex
                int* d_neighnum,//number of del neighbors
                int Ncells,
                int xsize,
                int ysize,
                double boxsize,
                periodicBoundaries Box,
                Index2D ci,
                Index2D cli,
                int* d_fixlist,
                Index2D GPU_idx,
                int *maximumNeighborNum,
                int currentMaxNeighborNum
                )
    {
    unsigned int block_size = THREADCOUNT;
    if (Ncells < THREADCOUNT) block_size = 32;
    unsigned int nblocks  = Ncells/block_size + 1;
    gpu_get_neighbors_no_sort_kernel<<<nblocks,block_size>>>(
                      d_pt,d_cell_sizes,d_cell_idx,P_idx,P,Q,Q_rad,d_neighnum,Ncells,xsize,ysize,
                      boxsize,Box,ci,cli,d_fixlist,GPU_idx,maximumNeighborNum,currentMaxNeighborNum
                      );

    HANDLE_ERROR(cudaGetLastError());
#ifdef DEBUGFLAGUP
    cudaDeviceSynchronize();
#endif
    return cudaSuccess;
    };

bool gpu_get_neighbors(double2* d_pt, //the point set
                unsigned int* d_cell_sizes,//points per bucket
                int* d_cell_idx,//cellListIdxs
                int* P_idx,//index of Del Neighbors
                double2* P,//location del neighborPositions
                double2* Q,//voronoi vertex positions
                double* Q_rad,//radius? associated with voro vertex
                int* d_neighnum,//number of del neighbors
                int Ncells,
                int xsize,
                int ysize,
                double boxsize,
                periodicBoundaries Box,
                Index2D ci,
                Index2D cli,
                int* d_fixlist,
                int Nf,
                Index2D GPU_idx,
                int *maximumNeighborNum,
                int currentMaxNeighborNum,
                bool globalRoutine
                )
    {
    unsigned int block_size = 64;
    if (Nf < 64) block_size = 32;
    unsigned int nblocks  = Nf/block_size + 1;

    /*
    Note: currently this is a templatized work-around for having fixed array sizes for Hv in the one_ring function
    One option would be to create a new data structure to store the ints of Hv of size N*MaxSize, etc.,
    or think of something more clever
    */

    if(globalRoutine)
        {
        gpu_get_neighbors_global_kernel<<<nblocks,block_size>>>(
                      d_pt,d_cell_sizes,d_cell_idx,P_idx,P,Q,Q_rad,d_neighnum,
                      Ncells,xsize,ysize,boxsize,Box,ci,cli,GPU_idx,maximumNeighborNum,currentMaxNeighborNum
                      );
        }
    else
        {
        gpu_get_neighbors_kernel<<<nblocks,block_size>>>(
                      d_pt,d_cell_sizes,d_cell_idx,P_idx,P,Q,Q_rad,d_neighnum,Ncells,xsize,ysize,
                      boxsize,Box,ci,cli,d_fixlist,Nf,GPU_idx,maximumNeighborNum,currentMaxNeighborNum
                      );
        }


    HANDLE_ERROR(cudaGetLastError());
#ifdef DEBUGFLAGUP
    cudaDeviceSynchronize();
#endif
    return cudaSuccess;
    };

bool gpu_build_repair(int* d_repair,
                 int Np,
                 int* Nf
                 )
    {
    unsigned int block_size = THREADCOUNT;
    if (Np < THREADCOUNT) block_size = 32;
    unsigned int nblocks  = Np/block_size + 1;

    unsigned int N=Np+1;
    thrust::sort(thrust::device,d_repair, d_repair + N);

    gpu_size_kernel<<<nblocks,block_size>>>(
                    d_repair,
                    Np,
                    Nf
                    );

    HANDLE_ERROR(cudaGetLastError());
#ifdef DEBUGFLAGUP
    cudaDeviceSynchronize();
#endif
    return cudaSuccess;
    };

//!call the kernel to test every circumcenter to see if it's empty
bool gpu_test_circumcenters(int *d_repair,
                            int3 *d_ccs,
                            int Nccs,
                            double2 *d_pt,
                            unsigned int *d_cell_sizes,
                            int *d_idx,
                            int Np,
                            int xsize,
                            int ysize,
                            double boxsize,
                            periodicBoundaries &Box,
                            Index2D &ci,
                            Index2D &cli
                            )
    {
    unsigned int block_size = THREADCOUNT;
    if (Nccs < THREADCOUNT) block_size = 32;
    unsigned int nblocks  = Nccs/block_size + 1;

    gpu_test_circumcenters_kernel<<<nblocks,block_size>>>(
                            d_repair,
                            d_ccs,
                            d_pt,
                            d_cell_sizes,
                            d_idx,
                            Nccs,
                            xsize,
                            ysize,
                            boxsize,
                            Box,
                            ci,
                            cli
                            );

    HANDLE_ERROR(cudaGetLastError());
#ifdef DEBUGFLAGUP
    cudaDeviceSynchronize();
#endif
    return cudaSuccess;
    };

/** @} */ //end of group declaration
