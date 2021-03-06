//**********************************************************
//Copyright 2011 Fethallah Benmansour
//
//Licensed under the Apache License, Version 2.0 (the "License"); 
//you may not use this file except in compliance with the License. 
//You may obtain a copy of the License at
//
//http://www.apache.org/licenses/LICENSE-2.0 
//
//Unless required by applicable law or agreed to in writing, software 
//distributed under the License is distributed on an "AS IS" BASIS, 
//WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
//See the License for the specific language governing permissions and 
//limitations under the License.
//**********************************************************


#ifndef __itkStructureTensorRecursiveGaussianImageFilter_h
#define __itkStructureTensorRecursiveGaussianImageFilter_h

#include "itkRecursiveGaussianImageFilter.h"
#include "itkNthElementImageAdaptor.h"
#include "itkImage.h"
#include "itkSymmetricSecondRankTensor.h"
#include "itkPixelTraits.h"
#include "itkProgressAccumulator.h"

namespace itk
{
  
  /** \class StructureTensorRecursiveGaussianImageFilter
   *
   * \brief Computes the structure tensor of a multidimensional image 
   * 
   * 
   * \ingroup GradientFilters   
   * \ingroup Singlethreaded
   */
  // NOTE that the ITK_TYPENAME macro has to be used here in lieu 
  // of "typename" because VC++ doesn't like the typename keyword 
  // on the defaults of template parameters
  template <typename TInputImage, 
  typename TOutputImage= Image< SymmetricSecondRankTensor< 
  typename NumericTraits< typename TInputImage::PixelType>::RealType,
  TInputImage::ImageDimension >,
  TInputImage::ImageDimension > >
  class ITK_EXPORT StructureTensorRecursiveGaussianImageFilter:
  public ImageToImageFilter<TInputImage,TOutputImage>
  {
  public:
    /** Standard class typedefs. */
    typedef StructureTensorRecursiveGaussianImageFilter  Self;
    typedef ImageToImageFilter<TInputImage,TOutputImage> Superclass;
    typedef SmartPointer<Self>                           Pointer;
    typedef SmartPointer<const Self>                     ConstPointer;
    
    
    /** Pixel Type of the input image */
    typedef TInputImage                                    InputImageType;
    typedef typename TInputImage::PixelType                PixelType;
    typedef typename NumericTraits<PixelType>::RealType    RealType;
    
    
    /** Image dimension. */
    itkStaticConstMacro(ImageDimension, unsigned int,
                        TInputImage::ImageDimension);
    
    /** Define the image type for internal computations 
     RealType is usually 'double' in NumericTraits. 
     Here we prefer float in order to save memory.  */
    
    typedef float                                            InternalRealType;
    typedef Image<InternalRealType, 
    itkGetStaticConstMacro(ImageDimension) >   RealImageType;
    
    /**  Output Image Nth Element Adaptor
     *  This adaptor allows to use conventional scalar 
     *  smoothing filters to compute each one of the 
     *  components of the gradient image pixels. */
    typedef NthElementImageAdaptor< TOutputImage,
    InternalRealType >  OutputImageAdaptorType;
    
    typedef typename OutputImageAdaptorType::Pointer OutputImageAdaptorPointer;
    
    /**  Smoothing filter type */
    typedef RecursiveGaussianImageFilter<
    RealImageType,
    RealImageType
    >    GaussianFilterType;
    
    /**  Derivative filter type, it will be the first in the pipeline  */
    typedef RecursiveGaussianImageFilter<
    InputImageType,
    RealImageType
    >    DerivativeFilterType;
    
    
    /**  Pointer to a gaussian filter. */
    typedef typename GaussianFilterType::Pointer    GaussianFilterPointer;
    
    /**  Pointer to a derivative filter. */
    typedef typename DerivativeFilterType::Pointer  DerivativeFilterPointer;
    
    /**  Pointer to the Output Image */
    typedef typename TOutputImage::Pointer          OutputImagePointer;
    
    
    /** Type of the output Image */
    typedef TOutputImage                                      OutputImageType;
    typedef typename          OutputImageType::PixelType      OutputPixelType;
    typedef typename PixelTraits<OutputPixelType>::ValueType  OutputComponentType;
    
    /** Method for creation through the object factory. */
    itkNewMacro(Self);
    
    /** Set Sigma value. Sigma is measured in the units of image spacing. */
    void SetSigma( RealType sigma );
    void SetRho( RealType rho );
    
    /** Define which normalization factor will be used for the Gaussian */
    void SetNormalizeAcrossScale( bool normalizeInScaleSpace );
    itkGetConstMacro( NormalizeAcrossScale, bool );
    itkGetConstMacro( Sigma, RealType );
    itkGetConstMacro( Rho,   RealType );
    
    /** StructureTensorRecursiveGaussianImageFilter needs all of the input to produce an
     * output. Therefore, StructureTensorRecursiveGaussianImageFilter needs to provide
     * an implementation for GenerateInputRequestedRegion in order to inform
     * the pipeline execution model.
     * \sa ImageToImageFilter::GenerateInputRequestedRegion() */
    virtual void GenerateInputRequestedRegion() throw(InvalidRequestedRegionError);
    
  protected:
    StructureTensorRecursiveGaussianImageFilter();
    virtual ~StructureTensorRecursiveGaussianImageFilter() {};
    void PrintSelf(std::ostream& os, Indent indent) const;
    
    /** Generate Data */
    void GenerateData( void );
    
    // Override since the filter produces the entire dataset
    void EnlargeOutputRequestedRegion(DataObject *output);
    
  private:
    StructureTensorRecursiveGaussianImageFilter(const Self&); //purposely not implemented
    void operator=(const Self&); //purposely not implemented
    
    RealType                                   m_Sigma;
    RealType                                   m_Rho;
    
    std::vector<GaussianFilterPointer>         m_SmoothingFilters;
    std::vector<GaussianFilterPointer>         m_TensorComponentSmoothingFilters;
    DerivativeFilterPointer                    m_DerivativeFilter;
    OutputImageAdaptorPointer                  m_ImageAdaptor;
    
    /** Normalize the image across scale space */
    bool m_NormalizeAcrossScale; 
    
  };
  
} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkStructureTensorRecursiveGaussianImageFilter.txx"
#endif

#endif
