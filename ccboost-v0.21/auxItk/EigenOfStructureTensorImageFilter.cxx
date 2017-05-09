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

#include <iostream>
#include <fstream>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>

#include <itkImage.h>
#include <itkIndex.h>
#include "itkImageFileWriter.h"
#include "itkImageFileReader.h"
#include "itkRescaleIntensityImageFilter.h"

#include "itkNthElementImageAdaptor.h"

#include "itkImageRegionIterator.h"

#include "itkStructureTensorRecursiveGaussianImageFilter.h"
#include "itkSymmetricEigenAnalysisImageFilter2.h"

#include "itkVectorIndexSelectionCastImageFilter.h"
#include "itkDivideImageFilter.h"
#include "itkMultiplyImageFilter.h"

#include "SymmetricEigenAnalysisTest.h"

// Macro to avoid re-typing with itk
// --> For instance: 
//			makeNew( random, itk::RandomImageSource<FloatImage2DType> );
// is equivalent to:
//			itk::RandomImageSource<FloatImage2DType>::Pointer	random;
//			random = itk::RandomImageSource<FloatImage2DType>::New();
// The __VA_ARGS__ is there so that it can handle commas within the argument list 
//   in a natural way
#define makeNew(instanceName, ...)	\
    __VA_ARGS__::Pointer instanceName = __VA_ARGS__::New()



//////////////////////////////////////////////////////////////////////
// I love typedefing my data

//Image
const int Dimension = 3;
typedef float                                     PixelType;
typedef itk::Image<PixelType, Dimension>          ImageType;
typedef ImageType::IndexType                      IndexType;
typedef itk::ImageFileReader< ImageType >         ReaderType;

typedef itk::FixedArray<float, Dimension>         OrientationPixelType;
typedef itk::Image<OrientationPixelType, Dimension>
OrientationImageType;
typedef itk::ImageFileReader< OrientationImageType >     OrientationImageReaderType;
typedef itk::ImageFileWriter< ImageType >     FloatImageWriterType;


typedef itk::ImageRegionConstIterator< OrientationImageType > ConstOrientationIteratorType;
typedef itk::ImageRegionIterator< OrientationImageType>       OrientationIteratorType;

typedef itk::ImageRegionConstIterator< ImageType > ConstFloatIteratorType;
typedef itk::ImageRegionIterator< ImageType>       FloatIteratorType;



/** Hessian & utils **/
typedef itk::SymmetricSecondRankTensor<float,Dimension>                       HessianPixelType;
typedef itk::Image< HessianPixelType, Dimension >                             HessianImageType;
typedef itk::StructureTensorRecursiveGaussianImageFilter<ImageType, HessianImageType> HessianFilterType;

typedef itk::Vector<float, Dimension>            VectorPixelType;
typedef itk::Vector<float, 2*Dimension>          EigenPixelType;

typedef itk::Image<VectorPixelType, Dimension> VectorImageType;
typedef itk::Vector<float, Dimension>          EigenValuePixelType;
typedef itk::Matrix<float, Dimension, Dimension>	EigenVectorPixelType;

typedef itk::Image<VectorPixelType, Dimension> VectorImageType;

typedef itk::Image<EigenValuePixelType, Dimension> EigenValueImageType;
typedef itk::Image<EigenVectorPixelType, Dimension> EigenVectorImageType;
typedef EigenValueImageType 		FirstEigenVectorOrientImageType;

typedef itk::SymmetricEigenAnalysisImageFilter2< HessianImageType, EigenValueImageType, EigenVectorImageType, FirstEigenVectorOrientImageType >        HessianToEigenFilter;

using namespace std;

//#define showMsg(args...) \
//    do { \
//        printf("\x1b[32m" "\x1b[1m["); \
//        printf(args); \
//        printf("]\x1b[0m\n" ); \
//    } while(0)

#define showMsg(args...) {}

int execute(float sigma, float rho, string imageName, string outputFile, bool sortByMagnitude, float zAnisotropyFactor)
{

            //  typename RotationalFeatureType::Pointer featureRotational
            //    = RotationalFeatureType::New();

            //////////////////////////////////////////////////////////////////////
            // Reads the image

    showMsg( "Reading the image" );

    ReaderType::Pointer reader = ReaderType::New();
    reader->SetFileName(imageName);
    reader->Update();
    
    
    ImageType::Pointer inpImg = reader->GetOutput();
    
    ImageType::SpacingType spacing = inpImg->GetSpacing();
    spacing[2] *= zAnisotropyFactor;
    
    //std::cout << "Using spacing: " << spacing << ", anisotr factor = " << zAnisotropyFactor << std::endl;
    inpImg->SetSpacing(spacing);
    
    

    showMsg("Hessian filtering");
    makeNew( hessianFilt, HessianFilterType );
    hessianFilt->SetSigma( sigma );
    hessianFilt->SetRho( rho );
    hessianFilt->SetInput( inpImg );

    showMsg("Computing eigenvalues");
    // now compute eigenvalues/main eigenvector
    makeNew( eigenFilt, HessianToEigenFilter );
    eigenFilt->SetInput( hessianFilt->GetOutput() );

    eigenFilt->SetOrderEigenMagnitudes(sortByMagnitude);
    eigenFilt->SetOrderEigenValues(!sortByMagnitude);
    eigenFilt->SetGenerateEigenVectorImage(false);
    eigenFilt->SetGenerateFirstEigenVectorOrientImage(false);

#if 0
    typedef itk::VectorResampleImageFilter< VectorImageType, VectorImageType > VectorResampleFilterType;
    makeNew( vecResampler, VectorResampleFilterType );

    vecResampler->SetInput(eigenFilt->GetOutput());
    vecResampler->SetSize(image->GetLargestPossibleRegion().GetSize());
#endif

    showMsg("Saving images");
    {
        eigenFilt->Update();
        EigenValueImageType* outImg = (EigenValueImageType *)eigenFilt->GetEigenValueImage();
        eigenFilt->Update();
        
        // reset spacing
        spacing[0] = spacing[1] = spacing[2] = 1.0;
        
        outImg->SetSpacing( spacing );
        
        makeNew( writer, itk::ImageFileWriter<EigenValueImageType> );
        writer->SetInput( outImg );
        writer->SetFileName( outputFile );
        writer->Update();
    }

    return EXIT_SUCCESS;
}


int main(int argc, char **argv)
{
    cjbCheckSymmetricEigenAnalysisEigMagnitudeOrdering();

    if(argc != 7){
        printf("Usage: EigenOfStructureTensorImageFilter image sigma rho zAnisotropyFactor outputFileWithEigenvalues sortByMagnitude\n");
        exit(0);
    }

    string imageName(argv[1]);

    float sigma  = atof(argv[2]);
    float rho = atof(argv[3]);
    string outputFile(argv[5]);
    float zAnisotropyFactor = atof(argv[4]);
    const bool sortByMagnitude = atoi(argv[6]) != 0;
	//printf("Order by magnitude: %d\n", sortByMagnitude );
	//printf("Sigma: %f, Rho: %f\n", sigma, rho );

	return execute(sigma, rho,  imageName, outputFile, sortByMagnitude, zAnisotropyFactor);
}
