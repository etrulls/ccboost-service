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
#include <itkVectorImage.h>
#include <itkIndex.h>
#include "itkImageFileWriter.h"
#include "itkImageFileReader.h"
#include "itkRescaleIntensityImageFilter.h"

#include "itkNthElementImageAdaptor.h"

#include "SymmetricEigenAnalysisTest.h"

#include "itkHessianRecursiveGaussianImageFilter.h"
#include "itkSymmetricEigenAnalysisImageFilter2.h"

#include "itkComposeImageFilter.h"
#include "itkMatrixIndexSelectionImageFilter.h"

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
typedef itk::HessianRecursiveGaussianImageFilter<ImageType, HessianImageType> HessianFilterType;

typedef itk::Vector<float, Dimension>            VectorPixelType;

typedef itk::Vector<float, Dimension>          EigenValuePixelType;
typedef itk::Matrix<float, Dimension, Dimension>	EigenVectorPixelType;

typedef itk::VectorImage<float, Dimension> VectorImageType;

typedef itk::Image<EigenValuePixelType, Dimension> EigenValueImageType;
typedef itk::Image<EigenVectorPixelType, Dimension> EigenVectorImageType;
typedef EigenValueImageType 		FirstEigenVectorOrientImageType;


typedef itk::ComposeImageFilter< ImageType >	ComposeFilterType;
typedef itk::MatrixIndexSelectionImageFilter< EigenVectorImageType, ImageType >	MatrixIndexSelectionFilterType;


typedef itk::SymmetricEigenAnalysisImageFilter2< HessianImageType, EigenValueImageType, EigenVectorImageType, FirstEigenVectorOrientImageType >        HessianToEigenFilter;

using namespace std;

#define showMsg(args...) \
    do { \
        printf("\x1b[32m" "\x1b[1m["); \
        printf(args); \
        printf("]\x1b[0m\n" ); \
    } while(0)

enum WhichEigVec
{
	EHighestMagnitude,
	ELowest, EHighest
};

int execute(float sigma, string imageName, string outputFile, WhichEigVec whichEig, float zAnisotropyFactor)
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
    
    std::cout << "Using spacing: " << spacing << ", anisotr factor = " << zAnisotropyFactor << std::endl;
    inpImg->SetSpacing(spacing);

    

    
    showMsg("Hessian filtering");
    makeNew( hessianFilt, HessianFilterType );
    hessianFilt->SetSigma( sigma );
    hessianFilt->SetInput( inpImg );

    showMsg("Computing eigenvalues");
    // now compute eigenvalues/main eigenvector
    makeNew( eigenFilt, HessianToEigenFilter );
    
    eigenFilt->SetGenerateEigenVectorImage(true);
    eigenFilt->SetGenerateFirstEigenVectorOrientImage(false);
    
    // only sort by magnitude if highest magnitude is required
    eigenFilt->SetOrderEigenValues( whichEig != EHighestMagnitude );
    eigenFilt->SetOrderEigenMagnitudes( whichEig == EHighestMagnitude );
    
    eigenFilt->SetInput( hessianFilt->GetOutput() );

#if 0
    typedef itk::VectorResampleImageFilter< VectorImageType, VectorImageType > VectorResampleFilterType;
    makeNew( vecResampler, VectorResampleFilterType );

    vecResampler->SetInput(eigenFilt->GetOutput());
    vecResampler->SetSize(image->GetLargestPossibleRegion().GetSize());
#endif

	makeNew( matSelFilt1, MatrixIndexSelectionFilterType );
	makeNew( matSelFilt2, MatrixIndexSelectionFilterType );
	makeNew( matSelFilt3, MatrixIndexSelectionFilterType );
	
	if ( whichEig == EHighestMagnitude || whichEig == EHighest )
    {
        matSelFilt1->SetIndices( 2, 0 );
        matSelFilt2->SetIndices( 2, 1 );
        matSelFilt3->SetIndices( 2, 2);
	} 
	else 
	{
        matSelFilt1->SetIndices( 0, 0 );
        matSelFilt2->SetIndices( 0, 1 );
        matSelFilt3->SetIndices( 0, 2 );
	}
	
	matSelFilt1->SetInput( eigenFilt->GetEigenVectorImage() );
	matSelFilt2->SetInput( eigenFilt->GetEigenVectorImage() );
	matSelFilt3->SetInput( eigenFilt->GetEigenVectorImage() );

    makeNew( composeFilt, ComposeFilterType );
    composeFilt->SetInput(0, matSelFilt1->GetOutput() );
    composeFilt->SetInput(1, matSelFilt2->GetOutput() );
    composeFilt->SetInput(2, matSelFilt3->GetOutput() );


    showMsg("Saving images");
    {
        composeFilt->Update();
        VectorImageType::Pointer outImg = composeFilt->GetOutput();
        composeFilt->Update();
        
        // reset spacing
        spacing[0] = spacing[1] = spacing[2] = 1.0;
        
        outImg->SetSpacing( spacing );
        
        
        makeNew( writer, itk::ImageFileWriter<VectorImageType> );
        writer->SetInput( outImg );
        writer->SetFileName( outputFile );
        writer->Update();
    }

    return EXIT_SUCCESS;
}


int main(int argc, char **argv)
{
    cjbCheckSymmetricEigenAnalysisEigMagnitudeOrdering();

    if(argc != 6){
        printf("Usage: RotationalFeaturesExtractFeatureVectors image sigma zAnisotropyFactor outputFile whichEig\n");
        printf("WhichEig:\n");
        printf("\tHighest magnitude: 0\n");
        printf("\tHighest: 1\n");
        printf("\tLowest: 2\n");
        exit(0);
    }

    string imageName(argv[1]);

    float sigma  = atof(argv[2]);
    float zAnisotropyFactor = atof(argv[3]);
    string outputFile(argv[4]);
    
    const int whichEigInt = atoi(argv[5]);
    WhichEigVec whichEig;
    switch(whichEigInt)
    {
		case 0:
			whichEig = EHighestMagnitude;
			break;
		case 1:
			whichEig = EHighest;
			break;
		case 2:
			whichEig = ELowest;
			break;
		default:
			printf("\tError in whichEig parameter\n");
			return -1;
			break;
	}


	return execute(sigma, imageName, outputFile, whichEig, zAnisotropyFactor);
}
