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

#include <itkGradientMagnitudeRecursiveGaussianImageFilter.h>

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
typedef unsigned char                             PixelType;
typedef itk::Image<PixelType, Dimension>          ImageType;
typedef ImageType::IndexType                      IndexType;
typedef itk::ImageFileReader< ImageType >         ReaderType;

typedef itk::Image<float, Dimension> OutputImageType;

typedef itk::ImageFileWriter< OutputImageType >     	  WriterType;


typedef itk::GradientMagnitudeRecursiveGaussianImageFilter<ImageType, OutputImageType> FilterType;

using namespace std;

#define showMsg(args...) \
    do { \
        printf("\x1b[32m" "\x1b[1m["); \
        printf(args); \
        printf("]\x1b[0m\n" ); \
    } while(0)

template < int Order, bool IncludeEven, bool IncludeOdd, bool Include0 >
int execute(float sigma, string imageName, string outputFile, float zAnisotropyFactor)
{
    showMsg( "Reading the image" );

    typename ReaderType::Pointer reader = ReaderType::New();
    reader->SetFileName(imageName);
    reader->Update();
    
    
    ImageType::Pointer inpImg = reader->GetOutput();
    
    ImageType::SpacingType spacing = inpImg->GetSpacing();
    spacing[2] *= zAnisotropyFactor;
    
    std::cout << "Using spacing: " << spacing << ", anisotr factor = " << zAnisotropyFactor << std::endl;
    inpImg->SetSpacing(spacing);


    showMsg("Filtering");
    makeNew( filter, FilterType );
    filter->SetSigma( sigma );
    filter->SetInput( inpImg );

#if 0
    typedef itk::VectorResampleImageFilter< VectorImageType, VectorImageType > VectorResampleFilterType;
    makeNew( vecResampler, VectorResampleFilterType );

    vecResampler->SetInput(eigenFilt->GetOutput());
    vecResampler->SetSize(image->GetLargestPossibleRegion().GetSize());
#endif

    {
        filter->Update();
        OutputImageType::Pointer outImg = filter->GetOutput();
        
        // reset spacing
        spacing[0] = spacing[1] = spacing[2] = 1.0;
        
        outImg->SetSpacing( spacing );
        
        makeNew( writer, WriterType );
        writer->SetInput( outImg );
        writer->SetFileName( outputFile );
        writer->Update();
    }

#if 0
    // save 1st eigval / 2nd eigval as well
    {
        makeNew( eigen1st, EigenElementSelectionFilter );
        eigen1st->SetIndex(0);    // highest eigenval
        eigen1st->SetInput(eigenFilt->GetOutput());

        makeNew( eigen2nd, EigenElementSelectionFilter );
        eigen2nd->SetIndex(1);    // 2nd highest eigenval
        eigen2nd->SetInput(eigenFilt->GetOutput());

        makeNew( eigen3rd, EigenElementSelectionFilter );
        eigen3rd->SetIndex(2);    // 2nd highest eigenval
        eigen3rd->SetInput(eigenFilt->GetOutput());

        if(1)
        {
            makeNew( divFilter, itk::DivideImageFilter< ImageType, ImageType, ImageType > );
            divFilter->SetInput1( eigen1st->GetOutput() );
            divFilter->SetInput2( eigen2nd->GetOutput() );

            makeNew(writer, itk::ImageFileWriter<ImageType>);
            writer->SetInput( eigen1st->GetOutput() );
            writer->SetFileName("/tmp/eigen1st.nrrd");
            writer->Update();
        }

        {
            makeNew(writer, itk::ImageFileWriter<ImageType>);
            writer->SetInput( eigen2nd->GetOutput() );
            writer->SetFileName("/tmp/eigen2nd.nrrd");
            writer->Update();
        }

        // now DoH
        {
            makeNew( mulFilter1, itk::MultiplyImageFilter< ImageType, ImageType, ImageType > );
            mulFilter1->SetInput1( eigen1st->GetOutput() );
            mulFilter1->SetInput2( eigen2nd->GetOutput() );

            makeNew( mulFilter2, itk::MultiplyImageFilter< ImageType, ImageType, ImageType > );
            mulFilter2->SetInput1( mulFilter1->GetOutput() );
            mulFilter2->SetInput2( eigen3rd->GetOutput() );

            makeNew(writer, itk::ImageFileWriter<ImageType>);
            writer->SetInput( mulFilter2->GetOutput() );
            writer->SetFileName("/tmp/DoH.nrrd");
            writer->Update();
        }

    }
#endif
    return EXIT_SUCCESS;
}


int main(int argc, char **argv) {

    if(argc != 5){
        printf("Usage: GaussianImageFilter image sigma zAnisotropyFactor outputFileWithEigenvalues\n");
        exit(0);
    }

    string imageName(argv[1]);
    float sigma  = atof(argv[2]);
    string outputFile(argv[4]);
    
    float zAnisotropyFactor = atof(argv[3]);

    execute<2,1,1,1>(sigma, imageName, outputFile, zAnisotropyFactor);

}
