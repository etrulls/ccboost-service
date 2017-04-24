//////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2013 Carlos Becker                                             //
// Ecole Polytechnique Federale de Lausanne                                     //
// Contact <carlos.becker@epfl.ch> for comments & bug reports                   //
//                                                                              //
// This program is free software: you can redistribute it and/or modify         //
// it under the terms of the version 3 of the GNU General Public License        //
// as published by the Free Software Foundation.                                //
//                                                                              //
// This program is distributed in the hope that it will be useful, but          //
// WITHOUT ANY WARRANTY; without even the implied warranty of                   //
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU             //
// General Public License for more details.                                     //
//                                                                              //
// You should have received a copy of the GNU General Public License            //
// along with this program. If not, see <http://www.gnu.org/licenses/>.         //
//////////////////////////////////////////////////////////////////////////////////

#ifndef INTEGRALIMAGE_H
#define INTEGRALIMAGE_H

#include <Matrix3D.h>
#include <vector>

// a box of center (x,y,z), radius (rx,ry,rz)
struct BoxPosition
{
    unsigned x,y,z;
    unsigned rx, ry, rz;
};

typedef std::vector<BoxPosition> BoxPositionList;


template<typename T>
class IntegralImage
{
private:
    Matrix3D<T>  mData;

public:

    // computes the integral image for src and saves results in dest
	//  src and dest can have != scalar types
    template<typename TSrc, typename TDest>
    static void staticCompute( Matrix3D<TDest> &dest, const Matrix3D<TSrc> &src )
    {
        dest.reallocSizeLike( src );


        for (unsigned z=0; z < src.depth(); z++)
        {
            TDest accum = 0;
            for (unsigned x=0; x < src.width(); x++) {
                accum += src(x,0,z);
                dest(x,0,z) = accum;
            }
        }


        for (unsigned z=0; z < src.depth(); z++)
        {
            for (unsigned y=1; y < src.height(); y++)
            {
                TDest yAccum = 0;
                for (unsigned x=0; x < src.width(); x++) {
                    yAccum += src(x,y,z);
                    dest(x,y,z) = yAccum + dest(x, y-1, z);
                }
            }
        }

        for (unsigned z=1; z < src.depth(); z++)
        {

            for (unsigned y=0; y < src.height(); y++)
            {
                for (unsigned x=0; x < src.width(); x++) {
                    //zAccum += ;
                    dest(x,y,z) += dest(x, y, z-1);
                }
            }
        }
    }

    IntegralImage() {}

    template<typename T2>
    void compute( const Matrix3D<T2> &m )
    {
        staticCompute( mData, m );
    }

    template<typename T2>
    void initializeToZero( const Matrix3D<T2> &sizeLike )
    {
        mData.reallocSizeLike( sizeLike );
        mData.fill(0);
    }

    inline T valAt(unsigned x, unsigned y, unsigned z) const
    {
        return mData(x,y,z);
    }

    // computes the sum between [x1,x2] [y1 y1] [z1 z2] inclusive
    // WARNING: doesn't check for negative values, (x1,y1,z2) cannot be less than (1,1,1)
    inline T volumeSum( unsigned int x1, unsigned int x2,
                        unsigned int y1, unsigned int y2,
                        unsigned int z1, unsigned int z2) const
    {
        x1--; y1--; z1--;
		
        return mData( x2, y2, z2 ) + mData( x2, y1, z1 ) + mData( x1, y2, z1 ) + mData( x1, y1, z2 )
                - mData( x2, y2, z1 ) - mData( x2, y1, z2 ) - mData( x1, y2, z2 ) - mData(x1, y1, z1);
    }

    // WARNING: doesn't check for negative values!
    inline T centeredSum( unsigned int x, unsigned int y, unsigned int z,
                          unsigned int rx, unsigned int ry, unsigned int rz ) const
    {
        return volumeSum( x - rx, x + rx,
                          y - ry, y + ry,
                          z - rz, z + rz );
    }


    // same as the other centeredSumNormalized() but subtracts mean and multiplies by inv Std
    inline T centeredSumNormalized( unsigned int x, unsigned int y, unsigned int z,
                                    unsigned int rx, unsigned int ry, unsigned int rz,
                                    T mean, T invStd) const
    {
		// approximating radius with a single coordinate!
        double fR = 2*rx + 1;

        return ( centeredSum(x,y,z,rx,ry,rz) / (fR * fR * fR) - mean) * invStd;
    }
    
#if 0
    // same as centeredSum() but normalized with the radius
    inline T centeredSumNormalized( unsigned int x, unsigned int y, unsigned int z, unsigned int r ) const
    {
#if 1
        double fR = 2*r + 1;
        return volumeSum( x - r, x + r,
                          y - r, y + r,
                          z - r, z + r ) / (fR * fR * fR);
#else
        return volumeSum( x - r, x + r,
                          y - r, y + r,
                          z - r, z + r );
#endif
    }
#endif

    inline T centeredSum( const BoxPosition &box ) const
    {
        //qDebug("Centered sum at: %d %d %d %d", (int)box.x, (int)box.y, (int)box.z, (int)box.r);
        return centeredSum( box.x, box.y, box.z, box.rx, box.ry, box.rz );
    }


    inline T centeredSumNormalized( const BoxPosition &box, T mean, T invStd ) const
    {
        return centeredSumNormalized( box.x, box.y, box.z, box.rx, box.ry, box.rz, mean, invStd );
    }
    
#if 0
    inline T centeredSumNormalized( const BoxPosition &box ) const
    {
        //qDebug("Centered sum at: %d %d %d %d", (int)box.x, (int)box.y, (int)box.z, (int)box.r);
        return centeredSumNormalized( box.x, box.y, box.z, box.r );
    }

    inline T centeredSumNormalized( const BoxPositionList &boxes ) const
    {
        const unsigned N = boxes.size();
        T accum = 0;
        for (unsigned i=0; i < N; i++)
            accum += centeredSumNormalized( boxes[i] );
        return accum;
    }
#endif
};

#endif // INTEGRALIMAGE_H
