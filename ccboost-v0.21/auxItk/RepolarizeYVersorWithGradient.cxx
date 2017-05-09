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

#include "itkImageFileWriter.h"
#include "itkImageFileReader.h"
#include "itkGradientRecursiveGaussianImageFilter.h"
#include "itkImage.h"

#include <itkImageRegionIteratorWithIndex.h>

/**
 * This filter will read an image + its eigenvector of hessian image
 * and for each pixel
 * 		1) do dot product between gradient vector and 2nd eigenvector
 * 		2) if the dot product is negative, invert 2nd eigenvector
 * 		3) save
 */

int main(int argc, char* argv[])
{  
   // Verify number of parameters in command line
  if( argc < 6 )
  {
    std::cerr << "Usage: " << std::endl;
    std::cerr << argv[0] << " inputImageFile eigVecImageFile sigma zAnisotropyFactor outputVectorImageFile " << std::endl;
    return EXIT_FAILURE;
  }
 
  //typedef
  typedef signed short          InputPixelType;
  typedef float                 ComponentType;
  const   unsigned int          Dimension = 3;
  
  const std::string  inputImgFile(argv[1]);
  const std::string  eigVecImgFile(argv[2]);
  const float sigma  = atof( argv[3] );
  const float zAnisotropyFactor  = atof( argv[4] );
  const std::string  outputImgFile(argv[5]);
  
  //printf("Sigma: %f\n", sigma);
 
  typedef itk::CovariantVector< ComponentType, 
								Dimension  >      GradientPixelType;
 
  typedef itk::Image< InputPixelType,  Dimension >    InputImageType;
  typedef itk::Image< GradientPixelType, Dimension >  GradientImageType;
  
  typedef itk::VectorImage< float, Dimension >			VectorImageType;
 
  typedef itk::ImageFileReader< InputImageType  >  ReaderType;
  typedef itk::ImageFileWriter< VectorImageType >  WriterType;
  
  typedef itk::ImageFileReader< VectorImageType  >  EigVecReaderType;
 
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName( inputImgFile  );
  reader->Update();
  
  // input image
    InputImageType::Pointer inpImg = reader->GetOutput();
    {
        InputImageType::SpacingType spacing = inpImg->GetSpacing();
        spacing[2] *= zAnisotropyFactor;

        //std::cout << "Using spacing: " << spacing << ", anisotr factor = " << zAnisotropyFactor << std::endl;
        inpImg->SetSpacing(spacing);
    }

  
  EigVecReaderType::Pointer eigvecReader = EigVecReaderType::New();
  eigvecReader->SetFileName( eigVecImgFile );
  eigvecReader->Update();
  
  // eigvec image
    VectorImageType::Pointer eigVecImg = eigvecReader->GetOutput();
    {
        VectorImageType::SpacingType spacing = eigVecImg->GetSpacing();
        spacing[2] *= zAnisotropyFactor;

        //std::cout << "Using spacing: " << spacing << ", anisotr factor = " << zAnisotropyFactor << std::endl;
        eigVecImg->SetSpacing(spacing);
    }
  
    
  // check that images are same size
  {
	  InputImageType::RegionType region1 = inpImg->GetLargestPossibleRegion();
	  InputImageType::RegionType region2 = eigVecImg->GetLargestPossibleRegion();
	  
	  if ((region1.GetSize()[0] != region2.GetSize()[0]) || (region1.GetSize()[1] != region2.GetSize()[1]) || (region1.GetSize()[2] != region2.GetSize()[2]))
	  {
		  printf("Region sizes don't match. Aborting..\n");
		  return -1;
	  }
	  
	  if (eigVecImg->GetNumberOfComponentsPerPixel() != 9)
	  {
		  printf("Components per pixel of eigvector image must be 9, %d found\n", eigVecImg->GetNumberOfComponentsPerPixel());
		  return -1;
	  }
  }
  
  //Filter class is instantiated
  typedef itk::GradientRecursiveGaussianImageFilter<InputImageType, GradientImageType> GradientFilterType;
 
  GradientFilterType::Pointer filter = GradientFilterType::New();
 
  filter->SetSigma( sigma );  
  filter->SetInput( inpImg );
  filter->Update();
  
  // let's do the processing manually and modify the image in eigvecReader
  VectorImageType::Pointer   evImg = eigVecImg;
  GradientImageType::Pointer grImg = filter->GetOutput();
  
  itk::ImageRegionIteratorWithIndex<VectorImageType> iter(evImg, evImg->GetLargestPossibleRegion());
  
  while(!iter.IsAtEnd())
  {
    GradientPixelType &grad = grImg->GetPixel( iter.GetIndex() );
    
    // no reference :(
    VectorImageType::PixelType ev = iter.Get();
    
    // dot product with 2nd eigvec
    float dotP = grad[0] * ev[3] + grad[1] * ev[4] + grad[2] * ev[5];
 
	if (dotP < 0)
	{
		// revert that eigvec
		ev[3] = -ev[3];
		ev[4] = -ev[4];
		ev[5] = -ev[5];
		
		iter.Set(ev);
	}
 
    ++iter;
  }
 
  
  // re-set spacing before writing to disk
  {
    VectorImageType::SpacingType spacing;
    spacing[0] = spacing[1] = spacing[2] = 1.0;
        
    evImg->SetSpacing( spacing );
  }
  
  
  // write to HD
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName(outputImgFile);
  writer->SetInput( evImg );
 
  try 
    { 
    //execute the pipeline
    writer->Update(); 
    } 
  catch( itk::ExceptionObject & err ) 
    { 
    std::cerr << "ExceptionObject caught !" << std::endl; 
    std::cerr << err << std::endl; 
    return EXIT_FAILURE;
    } 
 
  return EXIT_SUCCESS;
}
