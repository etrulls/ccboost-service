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

#include <Eigen/Dense>

static void bruteForceNormalSearch( const std::vector<Eigen::Matrix3f> &svOrient,
                                    const std::vector<unsigned> &SVList,
                                    Eigen::Vector3f *retNormal,
                                    unsigned *SVListIdx = 0,
                                    float *retCost = 0 )
{
    const unsigned N = SVList.size();

    // translate to eigen vectors, speed up (vectorization)
    Eigen::MatrixXf eOrients( 3, N );

    for (unsigned i=0; i < N; i++)
    {
        const unsigned idx = SVList[i];
        eOrients.coeffRef(0,i) = svOrient[idx].coeffRef(0,2);   // last column is z-orient
        eOrients.coeffRef(1,i) = svOrient[idx].coeffRef(1,2);
        eOrients.coeffRef(2,i) = svOrient[idx].coeffRef(2,2);
    }

    // compute dot products (assuming already normalized vectors)
    // a waste of space.. easier to code though
    unsigned maxValIdx = 0;
    float maxVal = ( eOrients.transpose() * eOrients ).cwiseAbs().rowwise().sum().maxCoeff( &maxValIdx );

    *retNormal = eOrients.col( maxValIdx );

    if (SVListIdx != 0)
        *SVListIdx = maxValIdx;

    if ( retCost != 0 ) *retCost = maxVal / N;
}
