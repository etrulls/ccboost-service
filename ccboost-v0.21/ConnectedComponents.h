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

#ifndef CONNECTEDCOMPONENTS_H
#define CONNECTEDCOMPONENTS_H

#include <itkImage.h>
#include <itkConnectedComponentImageFilter.h>
#include <itkBinaryThresholdImageFilter.h>
#include <Matrix3D.h>
#include <vector>

// class to extract connected components from GT image
template<typename tPixelType>
class ConnectedComponents
{
public:
    typedef tPixelType      PixelType;
    typedef unsigned int    LabelPixelType;

    typedef typename itk::Image<PixelType, 3>           ImageType;
    typedef typename itk::Image<LabelPixelType, 3>  	LabelImageType;

    typedef typename itk::BinaryThresholdImageFilter <ImageType, ImageType>
        BinaryThresholdImageFilterType;

    typedef typename itk::ConnectedComponentImageFilter< ImageType, LabelImageType >
        ConnectedCompFilterType;

private:
    typename LabelImageType::Pointer    mCCImage;
    unsigned int                        mNumCC;

    unsigned int mNumVoxels;

public:
    ConnectedComponents() { mCCImage = 0; mNumCC = 0; mNumVoxels = 0; }

    void process( const Matrix3D<tPixelType> &img, PixelType lowerThreshold, PixelType upperThreshold )
    {
        mNumVoxels = img.numElem();

        typename BinaryThresholdImageFilterType::Pointer thresholdFilter
                            = BinaryThresholdImageFilterType::New();
                    thresholdFilter->SetInput(img.asItkImage());
                    thresholdFilter->SetLowerThreshold(lowerThreshold);
                    thresholdFilter->SetUpperThreshold(upperThreshold);
                    thresholdFilter->SetInsideValue(255);
                    thresholdFilter->SetOutsideValue(0);


        // label connected components
        typename ConnectedCompFilterType::Pointer CCFilter = ConnectedCompFilterType::New();
                    CCFilter->SetInput( thresholdFilter->GetOutput() );
					//CCFilter->SetFullyConnected(true);

        CCFilter->Update();
        mCCImage = CCFilter->GetOutput();   // save pointer

        mNumCC = CCFilter->GetObjectCount();
    }

    void findRegionsBiggerThan( unsigned int minNumVoxels, std::vector< std::vector<unsigned int> > &retIdxList )
    {
        retIdxList.clear();

        const LabelPixelType *dataPtr = Matrix3D<LabelPixelType>::getItkImageDataPtr( mCCImage );

        for (unsigned int i=0; i < mNumCC; i++)
        {
            std::vector<unsigned int>   idxList;
            for (unsigned int p=0; p < mNumVoxels; p++)
                if ( dataPtr[p] == i + 1 )  idxList.push_back(p);

            if ( idxList.size() >= minNumVoxels ) {
                retIdxList.push_back( idxList );
            }
        }
    }
};

#endif // CONNECTEDCOMPONENTS_H
