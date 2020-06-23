#ifndef CUFUNCTIONS_H
#define CUFUNCTIONS_H

#include "std_include.h"
#include "gpuarray.h"
#include <set>
#include <algorithm>

#ifdef NVCC
/*!
\def HOSTDEVICE
__host__ __device__ inline
*/
#define HOSTDEVICE __host__ __device__ inline
#else
#define HOSTDEVICE inline __attribute__((always_inline))
#endif

/*! \file functions.h */

/** @defgroup Functions functions
 * @{
 \brief Utility functions that can be called from host or device
 */

//!Calculate the determinant of a 2x2 matrix
HOSTDEVICE double Det2x2(const double &x11,const double &x12, const double &x21, const double &x22)
    {
    return x11*x22-x12*x21;
    };

//!Calculates the circumcenter of the circle passing through x1, x2 and the origin
HOSTDEVICE void Circumcenter(const double2 &x1, const double2 &x2, double2 &xc)
    {
    double x1norm2,x2norm2,denominator;
    x1norm2 = x1.x*x1.x + x1.y*x1.y;
    x2norm2 = x2.x*x2.x + x2.y*x2.y;
    denominator = 0.5/(Det2x2(x1.x,x1.y,x2.x,x2.y));

    xc.x = denominator * Det2x2(x1norm2,x1.y,x2norm2,x2.y);
    xc.y = denominator * Det2x2(x1.x,x1norm2,x2.x,x2norm2);
    };

//!Calculates the circumcenter through x1,x2,x3
HOSTDEVICE void Circumcenter(const double2 &x1, const double2 &x2, const double2 &x3, double2 &xc)
    {
    double amcx,amcy,bmcx,bmcy,amcnorm2,bmcnorm2,denominator;

    amcx=x1.x-x3.x;
    amcy=x1.y-x3.y;
    amcnorm2 = amcx*amcx+amcy*amcy;
    bmcx=x2.x-x3.x;
    bmcy=x2.y-x3.y;
    bmcnorm2 = bmcx*bmcx+bmcy*bmcy;

    denominator = 0.5/(Det2x2(amcx,amcy,bmcx,bmcy));

    xc.x = x3.x + denominator * Det2x2(amcnorm2,amcy,bmcnorm2,bmcy);
    xc.y = x3.y + denominator * Det2x2(amcx,amcnorm2,bmcx,bmcnorm2);

    };

//!remove duplicate elements from a vector, preserving the order, using sets
template<typename T>
inline __attribute__((always_inline)) void removeDuplicateVectorElements(vector<T> &data)
    {
    set<T> currentElements;
    auto newEnd = std::remove_if(data.begin(),data.end(),[&currentElements](const T& value)
        {
        if (currentElements.find(value) != std::end(currentElements))
            return true;
        currentElements.insert(value);
        return false;
        });
    data.erase(newEnd,data.end());
    };

//!shrink a GPUArray by removing the i'th element and shifting any elements j > i into place
template<typename T>
inline __attribute__((always_inline)) void removeGPUArrayElement(GPUArray<T> &data, int index)
    {
    int n = data.getNumElements();
    GPUArray<T> newData;
    newData.resize(n-1);
    {//scope for array handles
    ArrayHandle<T> h(data,access_location::host,access_mode::read);
    ArrayHandle<T> h1(newData,access_location::host,access_mode::overwrite);
    int idx = 0;
    for (int i = 0; i < n; ++i)
        {
        if (i != index)
            {
            h1.data[idx] = h.data[i];
            idx += 1;
            };
        };
    };
    data = newData;
    };

//!shrink a GPUArray by removing the elements [i1,i2,...in] of a vector and shifting any elements j > i_i into place
template<typename T>
inline __attribute__((always_inline)) void removeGPUArrayElement(GPUArray<T> &data, vector<int> indices)
    {
    std::sort(indices.begin(),indices.end());
    int n = data.getNumElements();
    GPUArray<T> newData;
    newData.resize(n-indices.size());
    {//scope for array handles
    ArrayHandle<T> h(data,access_location::host,access_mode::read);
    ArrayHandle<T> h1(newData,access_location::host,access_mode::overwrite);
    int idx = 0;
    int vectorIndex = 0;
    for (int i = 0; i < n; ++i)
        {
        if (i != indices[vectorIndex])
            {
            h1.data[idx] = h.data[i];
            idx += 1;
            }
        else
            {
            vectorIndex += 1;
            if (vectorIndex >= indices.size())
                vectorIndex -= 1;
            };
        };
    };
    data = newData;
    };

//!print a double2 to screen
inline __attribute__((always_inline)) void printdouble2(double2 a)
    {
    printf("%f\t%f\n",a.x,a.y);
    };

//!grow a GPUArray, leaving the current elements the same but with extra capacity at the end of the array
template<typename T>
inline __attribute__((always_inline)) void growGPUArray(GPUArray<T> &data, int extraElements)
    {
    int n = data.getNumElements();
    GPUArray<T> newData;
    newData.resize(n+extraElements);
    {//scope for array handles
    ArrayHandle<T> h(data,access_location::host,access_mode::readwrite);
    ArrayHandle<T> hnew(newData,access_location::host,access_mode::overwrite);
    for (int i = 0; i < n; ++i)
        hnew.data[i] = h.data[i];
    };
    //no need to resize?
    data.swap(newData);
    };

//!fill the first data.size() elements of a GPU array with elements of the data vector
template<typename T>
inline __attribute__((always_inline)) void fillGPUArrayWithVector(vector<T> &data, GPUArray<T> &copydata)
    {
    int Narray = copydata.getNumElements();
    int Nvector = data.size();
    if (Nvector > Narray)
        copydata.resize(Nvector);
    ArrayHandle<T> handle(copydata,access_location::host,access_mode::overwrite);
    for (int i = 0; i < Nvector; ++i)
        handle.data[i] = data[i];
    };

//!copy a GPUarray to a vector
template<typename T>
inline __attribute__((always_inline)) void copyGPUArrayData(GPUArray<T> &data, vector<T> &copydata)
    {
    int n = data.getNumElements();
    ArrayHandle<T> handle(data,access_location::host,access_mode::read);
    copydata.resize(n);
    for (int i = 0; i < n; ++i) copydata[i] = handle.data[i];
    };

//!Get the circumcenter and radius, given one of the points on the circumcircle is the origin...
HOSTDEVICE void Circumcircle(const double2 &x1, const double2 &x2, double2 &xc, double &radius)
    {
    Circumcenter(x1,x2,xc);
    double dx = (x1.x-xc.x);
    double dy = (x1.y-xc.y);
    radius = sqrt(dx*dx+dy*dy);
    };

//!Get the circumcenter and radius given three distinct points on the circumcirle
HOSTDEVICE void Circumcircle(const double2 &x1, const double2 &x2, const double2 &x3, double2 &xc, double &radius)
    {
    Circumcenter(x1,x2,x3,xc);
    double dx = (x1.x-xc.x);
    double dy = (x1.y-xc.y);
    radius = sqrt(dx*dx+dy*dy);
    };

//!Helper function for the GPU DT to check if two line segments intersect
HOSTDEVICE bool ccw(const double2 &x1, const double2 &x2, const double2 &x3)
    {
    return (x3.y-x1.y)*(x2.x-x1.x) > (x2.y-x1.y)*(x3.x-x1.x);
    };

//!The dot product between two vectors of length two.
HOSTDEVICE double dot(const double2 &p1, const double2 &p2)
    {
    return p1.x*p2.x+p1.y*p2.y;
    };

//!The norm of a 2-component vector
HOSTDEVICE double norm(const double2 &p)
    {
    return sqrt(p.x*p.x+p.y*p.y);
    };

//!test if a point lies in an annulus, given the inner and outer radii
HOSTDEVICE bool inAnnulus(const double2 &p1, const double &rmin, const double &rmax)
    {
    double n = norm(p1);
    return (n > rmin && n < rmax);
    };

//!Calculate the area of a triangle with a vertex at the origin
HOSTDEVICE double SignedPolygonAreaPart(const double2 &p1, const double2 &p2)
    {
    return 0.5*(p1.x+p2.x)*(p2.y-p1.y);
    };


//!Calculate the area of a triangle with a vertex at the origin
HOSTDEVICE double TriangleArea(const double2 &p1, const double2 &p2)
    {
    return abs(0.5*(p1.x*p2.y-p1.y*p2.x));
    };

//!compute the sign of a double, and return zero if x = 0
HOSTDEVICE int computeSign(double x)
    {
    return ((x>0)-(x<0));
    };

//!compute the sign of a double, and return zero if x = 0...but return a double to avoid expensive casts on the GPU
HOSTDEVICE double computeSignNoCast(double x)
    {
    if (x > 0.) return 1.0;
    if (x < 0.) return -1.0;
    if (x == 0.) return 0.;
    return 0.0;
    };

//!Calculate which quadrant the point (x,y) is in
HOSTDEVICE int Quadrant(double x, double y)
    {
    if(x>=0)
        {
        if (y >=0)
            {return 0;
                }else{
                return  3;
                };
        }else{
        if (y>=0)
            {return 1;
                }else{
                return  2;
                };
        };
    //return -1;
    };


/** @} */ //end of group declaration

#undef HOSTDEVICE
#endif
