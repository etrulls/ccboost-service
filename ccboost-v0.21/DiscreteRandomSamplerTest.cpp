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

#include <cstdio>
#include "DiscreteRandomSampler.h"

int main()
{
    const unsigned N1 = 10;
    const unsigned N2 = 4;

    srand(time(NULL));

    Eigen::ArrayXd weights;
    weights.resize( N1 );
    weights << 0.2, 4, 0.9, 5, 7, 1, 6, 0.3, 0.1, 2;

    DiscreteRandomSampler< Eigen::ArrayXd > sampler;

    std::vector<unsigned int> idxs;


    for (int z=0; z < 1000; z ++)
    {
        sampler.sampleWithoutReplacement( weights, idxs, N2  );

        for (unsigned int i=0; i < N2; i++)
            printf("Idx %d: %.1f\n", idxs[i], weights[idxs[i]]);
        printf("\n");
    }

    return 0;
}
