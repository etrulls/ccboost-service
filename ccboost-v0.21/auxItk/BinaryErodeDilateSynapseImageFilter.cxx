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

#include "itkImage.h"
#include "itkBinaryErodeImageFilter.h"
#include "itkBinaryDilateImageFilter.h"
#include "itkImageFileReader.h"
#include "itkBinaryBallStructuringElement.h"

#include "itkAddImageFilter.h"
#include <itkMultiplyImageFilter.h>

#include "itkImageFileWriter.h"
 
const int Dimension = 3;

int main(int argc, char *argv[])
{
  if(argc != 5)
    {
    std::cerr << "Usage: " << std::endl;
    std::cerr << argv[0] << " InputImageFile erodeRadius dilateRadius OutputImageFile" << std::endl;
    return EXIT_FAILURE;
    }
 
    const int erodeRadius = atoi(argv[2]);
	const int dilateRadius = atoi(argv[3]);
  
  printf("Erode: %d, dilate: %d\n", erodeRadius, dilateRadius);

 
  typedef itk::Image<unsigned char, Dimension>    ImageType;
  typedef itk::ImageFileReader<ImageType> ReaderType;
  typedef itk::ImageFileWriter<ImageType> WriterType;
  
  typedef itk::MultiplyImageFilter< ImageType, ImageType, ImageType >  MultiplyFilterType;
  typedef itk::AddImageFilter< ImageType, ImageType, ImageType >  AddFilterType;
  
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName(argv[1]);
 
  typedef itk::BinaryBallStructuringElement<
    ImageType::PixelType, Dimension>                  StructuringElementType;
	
  StructuringElementType erodeElement, dilateElement;
  
  erodeElement.SetRadius(erodeRadius);
  erodeElement.CreateStructuringElement();
  
  dilateElement.SetRadius(dilateRadius);
  dilateElement.CreateStructuringElement();

  
  const unsigned char fgValue = 255;
  
  typedef itk::BinaryErodeImageFilter <ImageType, ImageType, StructuringElementType>
    BinaryErodeImageFilterType;
 
  BinaryErodeImageFilterType::Pointer erodeFilter
    = BinaryErodeImageFilterType::New();
	
  erodeFilter->SetInput(reader->GetOutput());
  erodeFilter->SetKernel(erodeElement);
  erodeFilter->SetErodeValue( fgValue );
  
  
  typedef itk::BinaryDilateImageFilter <ImageType, ImageType, StructuringElementType>
    BinaryDilateImageFilterType;
	
  BinaryDilateImageFilterType::Pointer dilateFilter
    = BinaryDilateImageFilterType::New();
	
  dilateFilter->SetInput(reader->GetOutput());
  dilateFilter->SetKernel(dilateElement);
  dilateFilter->SetDilateValue( fgValue );
  
  
  // now mix both, but first multiply
  MultiplyFilterType::Pointer mulErode = MultiplyFilterType::New();
  MultiplyFilterType::Pointer mulDilate = MultiplyFilterType::New();
  
  mulErode->SetInput( erodeFilter->GetOutput() );
  mulDilate->SetInput( dilateFilter->GetOutput() );
  
  mulErode->SetConstant( 127.0 / 255.0 );
  mulDilate->SetConstant( 128.0 / 255.0 );
  
  
  // now add
  AddFilterType::Pointer addFilt = AddFilterType::New();
  addFilt->SetInput1( mulDilate->GetOutput() );
  addFilt->SetInput2( mulErode->GetOutput() );
  

  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName(argv[4]);
  writer->SetInput( addFilt->GetOutput() );
  writer->Update();
 
  
 
  return EXIT_SUCCESS;
}
