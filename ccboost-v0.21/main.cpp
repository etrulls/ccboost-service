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

#include <sys/time.h>

#define qFatal(...) do { struct timeval tv; struct tm *tm; gettimeofday(&tv, NULL); tm = localtime(&tv.tv_sec); printf("%02d:%02d:%02d ", tm->tm_hour, tm->tm_min, tm->tm_sec); fprintf(stdout, "ERROR: "); fprintf (stdout, __VA_ARGS__); fprintf(stdout, "\n"); fflush(stdout); exit(-1); } while(0)
#define qPrint(...) do { struct timeval tv; struct tm *tm; gettimeofday(&tv, NULL); tm = localtime(&tv.tv_sec); printf("%02d:%02d:%02d ", tm->tm_hour, tm->tm_min, tm->tm_sec); fprintf (stdout, __VA_ARGS__); fprintf(stdout, "\n"); fflush(stdout); } while(0)
#define qDebug(...) {}

#define USE_POLARITY 0

#define LOCAL_ONLY 0

#define IGNORE_ORIENT_ESTIMATES 0

#define USE_MADABOOST   0

#define APPLY_PATCH_VARNORM 0

// if true, II is added which will contain predictions for
//  the last adaboost iteration at each supervoxel
#define USE_SPATIAL_BOOSTING 0

#define USE_AUTOCONTEXT 0

#if USE_AUTOCONTEXT
#define USE_SPATIAL_BOOSTING 1
#endif

// if orientation is interpreted as a rotation matrix
//   or as a single vector
#define USE_ALL_EIGVEC_FOR_ROTMATRICES  1

#define USE_LOGLOSS	0

// this can make a huge difference in mem usage
typedef double IntegralImageType;
//typedef float IntegralImageType;

// some macros to print float/double
template<typename T>
inline const char *  typeToString()
{
    return "UNKNOWN";
}

template<>
inline const char * typeToString<float>()
{
    return "Float";
}

template<>
inline const char * typeToString<double>()
{
    return "Double";
}

#include <algorithm>
#include <sstream>
#include <Matrix3D.h>
#include <slic/SuperVoxeler.h>
#include <cstdio>
#include <string>
#include <map>
#include <set>
#include <Eigen/Core>
#include "TimerRT.h"

#if USE_LOGLOSS
#include "LineSearch.h"
#endif

#include "HistogramMeanThreshold.h"
#include "DiscreteRandomSampler.h"
#include "ConnectedComponents.h"

#include "SigmoidFitterPlatt.h"

#include "itkBinaryErodeImageFilter.h"
#include "itkBinaryBallStructuringElement.h"

typedef unsigned char PixelType;

#include "ConfigData.h"
#include "EigenUtils.h"

#include "IntegralImage.h"
#include "SynapseUtils.hxx"


// sample M indices from 0..(N-1)
static void sampleWithoutReplacement( unsigned M, unsigned N, std::vector<unsigned> *idxs )
{
    if (M > N)  M = N;

    unsigned  max = N-1;

    std::vector<unsigned> toSample(N);
    for (unsigned i=0; i < N; i++)
        toSample[i] = i;

    idxs->resize(M);

    for (unsigned i=0; i < M; i++)
    {
        const unsigned idx = (((unsigned long)rand()) * max) / RAND_MAX;
        (*idxs)[i] = toSample[idx];

        //printf("Idx: %d / %d\n", idx, toSample[idx]);

        toSample[idx] = toSample[max];
        max = max - 1;
    }
}


// dummy func to convert something to string
template<typename T>
inline static std::string xToString( const T &val )
{
    std::stringstream strStream;
    strStream << val;
    return strStream.str();
}

// function to compute SV center
inline void computeSVCenter( const PixelInfoList &pList, UIntPoint3D *centroid )
{
    centroid->x = centroid->y = centroid->z = 0;
    for (unsigned int i=0; i < pList.size(); i++)
        centroid->add( pList[i].coords );

    centroid->divideBy( pList.size() );
}

#if !USE_ALL_EIGVEC_FOR_ROTMATRICES
void computeRotationMatrices( const std::vector<FloatPoint3D> &svOrient, std::vector< Eigen::Matrix3f > &rotMatrices )
{
    rotMatrices.resize( svOrient.size() );

    #pragma omp parallel for
    for (unsigned i=0; i < svOrient.size(); i++)
    {
        rotMatrices[i].col(2) << svOrient[i].x, svOrient[i].y, svOrient[i].z;

        // get the minimum
        int minIdx;
        rotMatrices[i].col(2).array().abs().minCoeff(&minIdx);

        Eigen::Vector3f tmpVec;
        tmpVec.setConstant(0);
        tmpVec(minIdx) = 1;

        rotMatrices[i].col(0) = rotMatrices[i].col(2).cross( tmpVec );

        // normalize
        rotMatrices[i].col(0) /= rotMatrices[i].col(0).norm();

        rotMatrices[i].col(1) = rotMatrices[i].col(2).cross( rotMatrices[i].col(0) );

        // debug output
        if (false)
        {
            if (i == 0)
            {
                std::cout << "\n";
                std::cout << "Rotation matrix test:" << std::endl;
                std::cout << "\tNorm0: " << rotMatrices[i].col(0).norm() << std::endl;
                std::cout << "\tNorm1: " << rotMatrices[i].col(1).norm() << std::endl;
                std::cout << "\tNorm2: " << rotMatrices[i].col(2).norm() << std::endl;
                std::cout << "\n";
            }
        }
    }
}
#endif

struct SVCombo
{
    SuperVoxeler<PixelType>     SVox;
    std::vector<UIntPoint3D>    svCentroids;

#if !USE_ALL_EIGVEC_FOR_ROTMATRICES
    // only if orientation is not needed
    std::vector<FloatPoint3D>       svOrient;
#endif
    std::vector< Eigen::Matrix3f >  rotMatrices;

    //Eigen::ArrayXXf histMatrix;

    // statistics, each vector corresponds to an integral image
    std::vector< std::vector<float> >  svoxWindowInvStd;
    std::vector< std::vector<float> >  svoxWindowMean;

    std::vector< IntegralImage<IntegralImageType> * > pixIntegralImages;

    Matrix3D<PixelType> rawImage;

    ~SVCombo()
    {
        for (unsigned i=0; i < pixIntegralImages.size(); i++)
            delete pixIntegralImages[i];
    }

    void invertRotMatrices()
    {
#if USE_ALL_EIGVEC_FOR_ROTMATRICES
        // we want to keep y-direction, but invert z
        //      so we have to invert x and z only
        for (unsigned int i=0; i < rotMatrices.size(); i++)
        {
            rotMatrices[i].col(0) = -rotMatrices[i].col(0);
            rotMatrices[i].col(2) = -rotMatrices[i].col(2);
        }
#else
        // invert and recompute
        for (unsigned int i=0; i < svCentroids.size(); i++)
        {
            svOrient[i].x = -svOrient[i].x;
            svOrient[i].y = -svOrient[i].y;
            svOrient[i].z = -svOrient[i].z;
        }
        computeRotationMatrices( svOrient, rotMatrices );
#endif
    }
};

template<typename WeakLearner>
class AdaBoost
{
public:
    typedef float                               FloatType;
    typedef Eigen::ArrayXd                      WeightsType;
    typedef typename WeakLearner::SampleIdxType SampleIdxType;

    typedef std::vector<SampleIdxType>          SampleIdxVector;    // idx
    typedef std::vector<unsigned int>           SampleClassVector;  // class (0/1/etc)
    typedef typename WeakLearner::ParamType     WeakLearnerParamType;

    typedef typename WeakLearner::SplitType     SplitType;

private:
    WeakLearner     mWeakLearner;

    SampleIdxVector     mSampleIdx;
    SampleClassVector   mSampleClass;

    // if it will search for the threshold on all the data after each iter
    bool mThresholdSearchOnWholeData;

    // contains a value !=0 if the given idx in mSampleIdx belongs to the minority class
    // so that it will always be sampled for training the weak learners
    std::vector<unsigned char>     mMinorityClassIdxs;

    WeightsType     mWeights;

    std::vector<SplitType>		mSplitList;
    std::vector<double>         mSplitWeightList;

    float       mStopErr;

    DiscreteRandomSampler<WeightsType>     mRandomSampler;

    // for Np positives, samplingRatio * Np negatives will be resampled
    unsigned    mReweightingSamplingRatio;

    unsigned int mRandSeed;
    std::vector<unsigned char>  mIsArtifNeg;

    // will save while learning, as a backup
    unsigned mBackupSaveLearnedStumpsEvery;

    // filename for backup stumps file
    std::string mBackupSavedLearnedStumpsFileName;

public:

    void setThresholdSearchOnWholeData( bool yes )
    {
        mThresholdSearchOnWholeData = yes;
    }

    void setBackupLearnedStumpsFileName( const std::string &fname )
    {
        mBackupSavedLearnedStumpsFileName = fname;
    }

    AdaBoost()
    {
        mRandSeed = 1234;
        mThresholdSearchOnWholeData = false;
        mBackupSaveLearnedStumpsEvery = 50;
    }

    const WeightsType & getSampleWeights() const
    {
        return mWeights;
    }

    const SampleIdxVector & getSampleIdxs() const
    {
        return mSampleIdx;
    }

    const SampleClassVector & getSampleClasses() const
    {
        return mSampleClass;
    }

    const std::vector<unsigned char> &getMinorityClassIdxs()
    {
        return mMinorityClassIdxs;
    }

    void reInit( const WeakLearnerParamType &wData )
    {
        mWeakLearner = wData;
    }

    unsigned numTrainedStumps() const
    {
        return mSplitList.size();
    }

    // only keeps the first 'max' stumps, removes the others
    void cropStumpsTo( unsigned max )
    {
        if ( max > mSplitList.size() )
            qFatal("Wanted to crop %d when there are only %d stumps", (int)max, (int) mSplitList.size() );

        const int toCrop = (int) mSplitList.size() - (int) max;

        for (int i=0; i < toCrop; i++)
        {
            mSplitList.pop_back();
            mSplitWeightList.pop_back();
        }
    }

    void setReweightingSamplingRatio( unsigned ratio )
    {
        mReweightingSamplingRatio = ratio;
    }


    void init( const WeakLearnerParamType &wData,
               const SampleIdxVector &samples,
               const SampleClassVector &sampleClasses )
    {
        mWeakLearner = wData;
        mSampleIdx = samples;
        mSampleClass = sampleClasses;

        mStopErr = 1e-6;
        mReweightingSamplingRatio = 10; // default

        if ( mSampleClass.size() != mSampleIdx.size() )
            qFatal("SampleClass and SampleIdx must have the same size");

        // by default set the minority class sampling to the positive ones
        std::vector<unsigned char>  minClassIdxs( mSampleIdx.size(), 0 );
        for (unsigned i=0; i < mSampleClass.size(); i++)
            if ( mSampleClass[i] == 1 ) minClassIdxs[i] = 1;

        setMinorityClassIdx( minClassIdxs );
    }

    void setArtifNegIdxs( const std::vector<unsigned char> &isArtifNeg )
    {
        mIsArtifNeg = isArtifNeg;
    }

    // this is for sampling, is taken as the idxs that are ALWAYS inside the dataset for weak learner training
    void setMinorityClassIdx( const std::vector<unsigned char> &minorityClassIdxs )
    {
        if ( minorityClassIdxs.size() != mSampleIdx.size() )
            qFatal("Minority class vector must be of same size as sample idx vector");

        mMinorityClassIdxs = minorityClassIdxs;
    }

    // if weightsType == null then it is not returned
    void resampleMinorityClass( SampleIdxVector &newSampleIdx,  SampleClassVector &newSampleClass, WeightsType *newWeights, std::vector<unsigned char> &isArtifNeg )
    {
        const unsigned samplingRatio = mReweightingSamplingRatio;

        std::vector<unsigned int>   alwaysIdxs;
        std::vector<unsigned int>   otherIdxs;
        for (unsigned int i=0; i < mSampleIdx.size(); i++)
        {
            if ( mMinorityClassIdxs[i] == 0 )
                otherIdxs.push_back(i);
            else
                alwaysIdxs.push_back(i);
        }

#define SAMPLE_HALF_BEFORE 1

#if SAMPLE_HALF_BEFORE
        {
            std::vector<unsigned> sampOther, sampAlways, sampUnlab;

            sampleWithoutReplacement( otherIdxs.size() / 2, otherIdxs.size(), &sampOther );
            sampleWithoutReplacement( alwaysIdxs.size() / 2, alwaysIdxs.size(), &sampAlways );


            std::vector<unsigned> A = alwaysIdxs;
            alwaysIdxs.clear();
            alwaysIdxs.resize( sampAlways.size() );
            std::vector<unsigned> O = otherIdxs;
            otherIdxs.clear();
            otherIdxs.resize( sampOther.size() );

            for (unsigned i=0; i < sampAlways.size(); i++)
                alwaysIdxs[i] = A[ sampAlways[i] ];

            for (unsigned i=0; i < sampOther.size(); i++)
                otherIdxs[i] = O[ sampOther[i] ];
        }
#endif


        const unsigned int numAlways = alwaysIdxs.size();
        const unsigned int numOther = otherIdxs.size();

        const unsigned int numOtherToSample = samplingRatio * numAlways;

        qDebug("Resampling minority class: [%d %d (%d)]", numAlways, numOtherToSample, samplingRatio);

        if (numOther < numOtherToSample)
            qFatal("Using sampling ratio %d with %d positive samples. Required negative examples: %d. Provided: %d", samplingRatio, numAlways, numOtherToSample, numOther);
            //qFatal("Num non-always assumed >= N * numAlwaysToSample!");

        newSampleIdx.resize(numAlways + numOtherToSample);
        newSampleClass.resize(numAlways + numOtherToSample);
        isArtifNeg.resize(numAlways + numOtherToSample);

        if (newWeights != 0)
            newWeights->resize(numAlways + numOtherToSample);

        //--- randomly sample according to weight
        {
            WeightsType otherWeights;
            otherWeights.resize( numOther );

            for (unsigned int i=0; i < numOther; i++)
                otherWeights.coeffRef(i) = mWeights.coeff( otherIdxs[i] );

            std::vector<unsigned int> subsampIdxs;
            mRandomSampler.sampleWithReplacement( otherWeights, subsampIdxs, numOtherToSample );

            for (unsigned int i=0; i < numOtherToSample; i++)
            {
                unsigned int z = otherIdxs[ subsampIdxs[i] ];
                newSampleIdx[i] = mSampleIdx[z];
                newSampleClass[i] = mSampleClass[ z ];
                isArtifNeg[i] = mIsArtifNeg[z];

                if (newWeights != 0)
                    newWeights->coeffRef(i) = mWeights.coeff( z );
            }

            if (newWeights != 0)
            {
                //qDebug("Max: %f / %f", otherWeights.maxCoeff(), newWeights->maxCoeff() );

                // set it to the mean, as in Geman & Fleuret
                newWeights->segment(0, numOtherToSample).setConstant( otherWeights.sum() / numOtherToSample );
            }

            //qDebug("Min/max other: %f %f", otherWeights.minCoeff(), otherWeights.maxCoeff());
        }


        // positives
        for (unsigned int i=0; i < numAlways; i++)
        {
            unsigned int z = alwaysIdxs[i];
            newSampleIdx[numOtherToSample + i] = mSampleIdx[z];
            newSampleClass[numOtherToSample + i] = mSampleClass[z];
            isArtifNeg[numOtherToSample + i] = mIsArtifNeg[z];

            if (newWeights != 0)
            {
                newWeights->coeffRef(numOtherToSample + i) = mWeights.coeff(z);
            }
        }

        *newWeights = (*newWeights) / (newWeights->sum());

        if (newWeights != 0)
            qDebug("Max: %f / %f / %f / sum: %f", mWeights.maxCoeff(), newWeights->maxCoeff(), newWeights->minCoeff(), newWeights->sum() );
    }

    void setRandomSeed( unsigned int newSeed )
    {
        mRandSeed = newSeed;
    }

#if USE_SPATIAL_BOOSTING
    // we need to keep track of these for spatial boosting
    Eigen::ArrayXf  mWholeImageScoreSoFar;
    std::vector<unsigned int>   mSVComboIndexes;

    static const unsigned mSpatBoostRecomputeEvery = 500;

    // returns true if it re-computed, so that a new iteration begins
    bool spatialBoostRecomputeII(SVCombo &svCombo, unsigned iterNumber)
    {
        const unsigned numSV = svCombo.SVox.numLabels();
        if (iterNumber == 0)
        {
            mWholeImageScoreSoFar.resize( numSV );
            mWholeImageScoreSoFar.fill(0);

            mSVComboIndexes.resize( numSV );
            for (unsigned i=0; i < numSV; i++)
                mSVComboIndexes[i] = i;

            return false;
        }

        // predict previous stump
        predictSingleStump( mSVComboIndexes, mWholeImageScoreSoFar, iterNumber - 1 );

        if ( (iterNumber % mSpatBoostRecomputeEvery ) != 0 )
            return false; // then keep the integral image the same

        // and assign to temporary image
        Matrix3D<float>  tempScoreImg;
        tempScoreImg.reallocSizeLike( svCombo.rawImage );

        for (unsigned i=0; i < numSV; i++)
        {
            const PixelInfoList &pixList = svCombo.SVox.voxelToPixel().at(mSVComboIndexes[i]);

            const float data = mWholeImageScoreSoFar.coeff(i);

            for (unsigned int p=0; p < pixList.size(); p++)
                tempScoreImg.data()[ pixList[p].index ] = data;
        }

        // recompute II
        svCombo.pixIntegralImages.back()->compute( tempScoreImg );
        return true;
    }

#endif

    // if reLearnThreshodls = true => expects splits already loaded
    //  svCombo needed for spatial boosting only
    bool learn( unsigned int numIters, SVCombo &svCombo, bool reLearnThresholds = false )
    {
        const unsigned int N = mSampleIdx.size();

        if (reLearnThresholds)
        {
            if ( numIters > mSplitList.size() )
                qFatal("Requested to relearn thresholds but numIter = %u and split num = %u", numIters, (unsigned int)mSplitList.size());
        }

        qPrint("Starting learning phase, seed: %u", mRandSeed);

        srand ( mRandSeed );
        mRandomSampler.reSeed( mRandSeed );

#if USE_LOGLOSS
        // we need to store the current prediction function value
        WeightsType	curPrediction(N);
        curPrediction.setZero();

        WeightsType labelVector(N);
        for (unsigned i=0; i < N; i++)
            labelVector.coeffRef(i) = (mSampleClass[i] == 1) ? 1 : (-1);
#endif

        // initialize weights
        mWeights.resize( N );
        mWeights.setConstant(1.0);
        mWeights = mWeights / mWeights.sum();

        // normalize wrt num of samples
        if (true)
        {
            qPrint("Normalizing weights.");
            //count number of pos
            unsigned numPos = 0;
            unsigned numNeg = 0;
            for (unsigned i=0; i < mSampleClass.size(); i++)
                if (mSampleClass[i] == 1)
                    numPos++;
                else
                    numNeg++;

            const double negWeight = 1.0 / numNeg;
            const double posWeight = 1.0 / numPos;

            for (unsigned i=0; i < mSampleClass.size(); i++)
                mWeights.coeffRef(i) = (mSampleClass[i] == 1) ? posWeight : negWeight;

            mWeights = mWeights / mWeights.sum();
        }

        WeightsType origWeights = mWeights;


        Eigen::ArrayXf  prediction;	// prediti
        Eigen::ArrayXf  scoreSoFar;

        TimerRT learnerTimer;
        double samplingTime, wlSearchTime, updateTime;

        const WeightsType initialWeights = mWeights;

        for (unsigned int i=0; i < numIters; i++)
        {
            learnerTimer.Reset();
            double lastTime = learnerTimer.elapsed();

#if USE_SPATIAL_BOOSTING
            qDebug("Evaluating scores for spatial boost");

            const bool spatBoostHappened = spatialBoostRecomputeII( svCombo, i );
#if USE_AUTOCONTEXT
            if (spatBoostHappened)
            {
                // need to reset scores, weights, etc
#if USE_LOGLOSS
                curPrediction.setZero();
#endif

                mWeights = initialWeights;
            }
#endif

            qDebug("End spatial boost");
#endif

            // split
            WeightsType newWeights;
            SampleIdxVector newSampleIdx;
            std::vector<unsigned> newSampleClass;

            std::vector<unsigned char> newIsArtifNeg;

            resampleMinorityClass( newSampleIdx, newSampleClass, &newWeights, newIsArtifNeg );

            samplingTime = learnerTimer.elapsed() - lastTime;
            lastTime = samplingTime;

            SplitType split;

            AdaBoostErrorType weakErr = 0;
            if (!reLearnThresholds)
                weakErr = mWeakLearner.learn( newSampleIdx, newSampleClass, newWeights, split );
            else
            {
                weakErr = mWeakLearner.learn( newSampleIdx, newSampleClass, newWeights, split, &mSplitList[i] );
            }
            /*AdaBoostErrorType weakErr = 0.1;
            split = SplitType( 16, 0.83, false, 162 );*/
            //split = SplitType( 1, 0.83, false, 162 );

            wlSearchTime = learnerTimer.elapsed() - lastTime;
            lastTime = wlSearchTime;

            //AdaBoostErrorType err = mWeakLearner.learn( mSampleIdx, mSampleClass, mWeights, split );
#ifdef _OPENMP
            const unsigned int numThreads = omp_get_max_threads();
#else
            const unsigned int numThreads = 1;
#endif
retry:
            AdaBoostErrorType err = 0;
            split.template classify< std::vector<unsigned>, (int)(-1) > ( mSampleIdx,  mWeakLearner.params(), prediction, numThreads );
            {
                for (unsigned int j=0; j < N; j++)
                {
                    if ( (prediction.coeff(j) > 0) != (mSampleClass[j] == 1) )
                        err += mWeights(j);
                }
            }

            if ( err > 0.5 )
            {
                qDebug("Error > 0.5, breaking... (%f %f)", err, weakErr);
                split.invertClassifier();
                goto retry;
                //break;
            }

            // re-learn threshold?
            if (mThresholdSearchOnWholeData)
            {
                // get feature value (without thresholding)
                Eigen::ArrayXf featVal( mSampleIdx.size() );
                split.template exportFeat< std::vector<unsigned>, Eigen::ArrayXf >( mSampleIdx, mWeakLearner.params(), featVal, 0 );

                // get best threshold / invert
                typedef FeatureOperatorPrecomputedValues< Eigen::ArrayXf, WeightsType > FeatureOpType;

                bool inv;
                IntegralImageType thr;

                err = findBestThreshold<FeatureOpType, false>( FeatureOpType( featVal, mWeights, mSampleClass, mSampleIdx ),
                        thr, inv );
                qDebug("Prev thr/inv: %f %d", split.threshold(), split.invert());
                split.setInvert( inv );
                split.setThreshold(thr);
                qDebug("New thr/inv: %f %d", split.threshold(), split.invert());

                // and re-predict (TODO; this can be avoided easily!)
                split.template classify< std::vector<unsigned>, (int)(-1) > ( mSampleIdx,  mWeakLearner.params(), prediction );
            }


#if USE_LOGLOSS
            // line search
            double alpha = 0;
            {
                WeightsType predTemp = prediction.cast<double>();
                LineSearch<WeightsType, WeightsType>	LS( curPrediction, predTemp, labelVector, LogLoss, initialWeights );
                alpha = LS.run();

                //alpha *= 0.1; //shrinkage?

                // update current prediction
                curPrediction += alpha * predTemp;

                // set weights
                //mWeights = (-2 * curPrediction * labelVector).exp() / ( 1 + (-2 * curPrediction * labelVector).exp() );
                mWeights = ((-2 * curPrediction * labelVector).exp() * initialWeights) / ( 1 + (-2 * curPrediction * labelVector).exp() ).square();
            }
#endif

#if !USE_LOGLOSS
            double alpha = 0.5 * log( (1.0 - err) / err );

            //alpha *= 0.1; //shrinkage?

            if (reLearnThresholds)
                alpha = mSplitWeightList[i];

            double expPlus = exp( alpha );
            double expNeg = exp( -alpha );

            for (unsigned int j=0; j < N; j++)
            {
                if ( (prediction.coeff(j) > 0) == (mSampleClass[j] == 1) )
                    mWeights.coeffRef(j) *= expNeg;
                else
                    mWeights.coeffRef(j) *= expPlus;


#if USE_MADABOOST
                if (mWeights.coeffRef(j) > origWeights.coeffRef(j))
                    mWeights.coeffRef(j) = origWeights.coeffRef(j);
#endif

#if USE_MADABOOST_ARTIFNEG
                if(mIsArtifNeg[j])
                    if (mWeights.coeffRef(j) > origWeights.coeffRef(j))
                        mWeights.coeffRef(j) = origWeights.coeffRef(j);
#endif
            }
#endif

            // normalize
            double wSum = mWeights.sum();
            mWeights = mWeights / wSum;
#if USE_MADABOOST || USE_MADABOOST_ARTIFNEG
            origWeights = origWeights / wSum;
#endif

            if (!reLearnThresholds)
            {
                mSplitList.push_back( split );
                mSplitWeightList.push_back( alpha );
            }
            else
            {
                // only modify split, keep threshold
                mSplitList[i] = split;
            }

            // compute overall error
            AdaBoostErrorType overallErr = 0;
            if(1)
            {
                predictSingleStump( mSampleIdx, scoreSoFar, i, numThreads );

                unsigned int errCount = 0;
                for (unsigned int i=0; i < mSampleIdx.size(); i++)
                {
                    if ( scoreSoFar.coeff(i) > 0 && mSampleClass[i] == 0 )
                        errCount++;
                    if ( scoreSoFar.coeff(i) < 0 && mSampleClass[i] != 0 )
                        errCount++;
                }
                overallErr = errCount * 1.0 / N;
            }

            qDebug( "%s", split.getStringDescription().c_str() );
            qDebug("------->   Iter %.3d: Error: %f (%f / %f this iter)", i, overallErr, err, weakErr);

            {
                updateTime = learnerTimer.elapsed() - lastTime;
                qDebug("Detailed timing: %f\t%f\t%f", samplingTime, wlSearchTime, updateTime);

                double elapsedTime = learnerTimer.elapsed();
                qDebug("Took: %.2f sec, Estimated left: %.2f hr",
                       elapsedTime,
                       (numIters - i) * elapsedTime / 3600.0);

                // Summarize output
                float t_est = (numIters - i) * elapsedTime;
                int t_h = floor(t_est / 3600.);
                t_est -= 3600. * t_h;
                int t_m = floor(t_est / 60.);
                t_est -= 60. * t_m;
                int t_s = round(t_est);
                //if((i+1) % 10 == 0)
                qPrint("Training: iter %d, est. left: %d:%02d:%02d", i, t_h, t_m, t_s);
            }

            /*if ( overallErr < mStopErr ) {
                qDebug("Stopping error reached!");
                break;
            }*/

            if ((i % mBackupSaveLearnedStumpsEvery) == 1)
            {
                qPrint("Saving backup stumps");
                saveLearnedStumps( mBackupSavedLearnedStumpsFileName, 0, i + 1 );
            }
        }

        //qDebug("Final error: %.2f", err * 1.0 / N);

        return true;
    }

    // saves learned stumps to a given file + platt scaling
    // if platt == 0 => default values are saved
    // if numStumps == 0 => all are saved
    bool saveLearnedStumps( const std::string &fName, const SigmoidFitterPlatt<double> *platt, unsigned numStumpsToSave = 0 ) const
    {
        if (mSplitList.size() == 0)
        {
            qPrint("Warning: classifier not saved, nothing to save!");
            return false;
        }

        if (mSplitList.size() < numStumpsToSave)
            numStumpsToSave = mSplitList.size();

        if (numStumpsToSave == 0)
            numStumpsToSave = mSplitList.size();

        libconfig::Config cfg;
        libconfig::Setting &root = cfg.getRoot();

        libconfig::Setting &stumps = root.add("stumps", libconfig::Setting::TypeList);

        for (unsigned i=0; i < numStumpsToSave; i++)
        {
            libconfig::Setting &st = stumps.add(libconfig::Setting::TypeGroup);

            mSplitList[i].save( st );

            st.add("abweight", libconfig::Setting::TypeFloat) = mSplitWeightList[i];
        }

        libconfig::Setting &st = root.add("plattScaling", libconfig::Setting::TypeGroup);
        if( platt != 0)
        {
            platt->save( st );
        }
        else
        {
            SigmoidFitterPlatt<double> blankPlatt;
            blankPlatt.save( st );
        }

        try
        {
            cfg.writeFile(fName.c_str());
            //std::cout << "Stumps successfully written to: " << fName
            //     << std::endl;

        }
        catch(const libconfig::FileIOException &fioex)
        {
            std::cerr << "I/O error while writing file: " << fName << std::endl;
            return false;
        }

        return true;
    }

    bool loadStumps( const std::string &fName, SigmoidFitterPlatt<double> &platt )
    {
        libconfig::Config cfg;
        cfg.readFile( fName.c_str() );

        libconfig::Setting &st = cfg.lookup("stumps");

        mSplitList.clear();
        mSplitWeightList.clear();
        for (unsigned i=0; i < (unsigned) st.getLength(); i++)
        {
            mSplitList.push_back( SplitType() );
            mSplitWeightList.push_back(0);

            mSplitList.back().load( st[i] );
            st[i].lookupValue( "abweight", mSplitWeightList.back() );
        }

        platt.load( cfg.lookup("plattScaling") );

        qPrint("Loaded %d stumps", (int) mSplitList.size() );

        return true;
    }

    // accumulates the single stump prediction in scoreAccum
    void predictSingleStump( const SampleIdxVector &samples, Eigen::ArrayXf  &scoreAccum, unsigned int stumpIdx, unsigned numThreads = 1  )
    {
        if (stumpIdx >= mSplitList.size())
            qFatal("Invalid stumpIdx");

        if (scoreAccum.rows() != (int)samples.size())
        {
            scoreAccum.resize( samples.size() );
            scoreAccum.setConstant(0);
        }

        std::vector<float> prediction( samples.size() );
        Eigen::Map< Eigen::ArrayXf >  predEigen( prediction.data(), prediction.size() );

        mSplitList[stumpIdx].classify( samples, mWeakLearner.params(), prediction, numThreads );

        scoreAccum += mSplitWeightList[stumpIdx] * (predEigen * 2 - 1);
    }

    void printSplitInformation()
    {
        for (unsigned int i=0; i < mSplitList.size(); i++)
        {
            qDebug("Stump %.3d: %.3f %s",
                   i,
                   mSplitWeightList[i],
                   mSplitList[i].getStringDescription().c_str());
        }
    }

    void predict( const SampleIdxVector &samples, Eigen::ArrayXf  &score, SVCombo &svCombo, bool normalizePrediction = false )
    {
        score.resize( samples.size() );
        score.setConstant(0);
#if USE_SPATIAL_BOOSTING == 0
        #pragma omp parallel for
#endif
        for (unsigned int i=0; i < mSplitList.size(); i++)
        {
#if USE_SPATIAL_BOOSTING
            const bool spatBoostHappened = spatialBoostRecomputeII(svCombo, i);
#if USE_AUTOCONTEXT
            if (spatBoostHappened)
                score.setConstant(0);	//reset
#endif
#endif

            Eigen::ArrayXf predEigen( samples.size() );

            mSplitList[i].template classify<SampleIdxVector, -1>( samples, mWeakLearner.params(), predEigen );

#if USE_SPATIAL_BOOSTING == 0
            #pragma omp critical
#endif
            score += mSplitWeightList[i] * predEigen;
        }

        //qDebug("Max / Min: %f  %f", score.maxCoeff(), score.minCoeff());

        if (normalizePrediction)
        {
            double Z = 0;
            for (unsigned i=0; i < mSplitWeightList.size(); i++)
                Z += mSplitWeightList[i];

            score /= Z;
            //qDebug("Z: %f", Z);
            //qDebug("After Max / Min: %f  %f", score.maxCoeff(), score.minCoeff());
        }
    }

    void exportFeatures( const SampleIdxVector &samples, Eigen::ArrayXXf &dest )
    {
        dest.resize( samples.size(), mSplitList.size() );
        for (unsigned int i=0; i < mSplitList.size(); i++)
        {
            mSplitList[i].exportFeat( samples, mWeakLearner.params(), dest, i );
        }
    }
};


// save file names
template<typename T>
bool  LoadSVox( const std::string &volFileName, int step, int cubeness, SuperVoxeler<T> &svox )
{
    char fName[1024];

    sprintf( fName, "%s-supervoxel-%d-%d.nrrd", volFileName.c_str(), step, cubeness );

    // try to load file
    if (!svox.load( fName ))
    {
        qPrint("Could not load SV cache");
        return false;
    }

    return true;
}

template<typename T>
bool SaveSVox( const std::string &volFileName, int step, int cubeness, const SuperVoxeler<T> &svox)
{
    char fName[1024];

    sprintf( fName, "%s-supervoxel-%d-%d.nrrd", volFileName.c_str(), step, cubeness );

    if (!svox.save( fName ))
    {
        qPrint("Could not save SV cache");
        return false;
    }

    return true;
}


#if APPLY_PATCH_VARNORM
// requires int image to be already loaded as well as centroids
template<typename T>
void computeSvoxWindowStatistics( SVCombo &combo, const unsigned iiIdx, const Matrix3D<T> &rawData )
{
    const unsigned numSV = combo.SVox.numLabels();

    IntegralImage<IntegralImageType> squaredII;
    {
        // temporary squared image
        Matrix3D<IntegralImageType> squaredImg;
        squaredImg.reallocSizeLike( rawData );

        for (unsigned i=0; i < rawData.numElem(); i++)
        {
            IntegralImageType val = rawData.data()[i];
            squaredImg.data()[i] = val*val;
        }

        // compute squared II
        squaredII.compute( squaredImg );
    }

    const int boxSize = 40;
    const int Vwidth = rawData.width();
    const int Vheight = rawData.height();
    const int Vdepth = rawData.depth();

    qPrint("Computing mean and std dev with box size = %d", boxSize);

    // reserve space
    combo.svoxWindowMean.push_back( std::vector<float>() );
    combo.svoxWindowInvStd.push_back( std::vector<float>() );

    std::vector<float> &svoxWindowMean = combo.svoxWindowMean.back();
    std::vector<float> &svoxWindowInvStd = combo.svoxWindowInvStd.back();

    svoxWindowMean.resize( numSV );
    svoxWindowInvStd.resize( numSV );

    #pragma omp parallel for
    for (unsigned i=0; i < numSV; i++)
    {
        int x = combo.svCentroids[i].x;
        int y = combo.svCentroids[i].y;
        int z = combo.svCentroids[i].z;

        // check image borders
        if ( x - boxSize <= 1 ) x = boxSize + 1;
        if ( y - boxSize <= 1 ) y = boxSize + 1;
        if ( z - boxSize <= 1 ) z = boxSize + 1;

        if ( x + boxSize >= Vwidth )   x = Vwidth - boxSize - 1;
        if ( y + boxSize >= Vheight)   y = Vheight - boxSize - 1;
        if ( z + boxSize >= Vdepth )   z = Vdepth - boxSize - 1;

        // TODO: fix boxSize for each coord
        IntegralImageType mean = combo.pixIntegralImages[iiIdx]->centeredSumNormalized( x, y, z, boxSize, boxSize, boxSize, 0, 1 );

        IntegralImageType stdDev = sqrt( squaredII.centeredSumNormalized( x, y, z, boxSize, boxSize, boxSize, 0, 1 ) - mean*mean );

#if 0
        IntegralImageType realStdDev = 0;
        // compare std dev vs real one
        for (unsigned qx=x - boxSize; qx <= x + boxSize; qx++)
            for (unsigned qy=y - boxSize; qy <= y + boxSize; qy++)
                for (unsigned qz=z - boxSize; qz <= z + boxSize; qz++)
                {
                    realStdDev += pow(combo.rawImage(qx,qy,qz) - mean, 2);
                }

        double fR = 2*boxSize + 1;
        realStdDev = sqrt(realStdDev / (fR * fR * fR));
        qDebug("Real vs computed: %f %f", realStdDev, stdDev);

#endif

        svoxWindowMean[i] = mean;
        svoxWindowInvStd[i] = 1.0 / (stdDev + 1e-6);
    }

    {
        Eigen::Map< Eigen::ArrayXf > meanArray( svoxWindowMean.data(), svoxWindowMean.size() );
        Eigen::Map< Eigen::ArrayXf > invStdArray( svoxWindowInvStd.data(), svoxWindowInvStd.size() );

        qPrint("Mean range:   %f -> %f", meanArray.minCoeff(), meanArray.maxCoeff());
        qPrint("InvStd range: %f -> %f", invStdArray.minCoeff(), invStdArray.maxCoeff());
    }
}
#endif

void appendFeatures( const std::vector<std::string> &featFNameArray,
                     const Matrix3D<unsigned char> &rawImage,
                     std::vector< IntegralImage<IntegralImageType> * > &IIs,
                     SVCombo &combo )
{
    typedef itk::VectorImage<float, 3>  ItkVectorImageType;

    // the num of threads must be limited to avoid mem overflow
    //#pragma omp parallel for ordered schedule(dynamic) num_threads(4)
    for (unsigned fNameIdx=0; fNameIdx < featFNameArray.size(); fNameIdx++)
    {
        const std::string &featFName = featFNameArray[fNameIdx];

        itk::ImageFileReader<ItkVectorImageType>::Pointer reader = itk::ImageFileReader<ItkVectorImageType>::New();
        try
        {
            reader->SetFileName( featFName );
            reader->Update();
        }
        catch(std::exception &e)
        {
            qFatal("Exception!");
        }

        /*if ( svCentroids.size() != SVox.numLabels() )
                qFatal("Centroids must == numlabels");*/

        ItkVectorImageType::Pointer img = reader->GetOutput();


        ItkVectorImageType::SizeType imSize = img->GetLargestPossibleRegion().GetSize();
        unsigned int mWidth = imSize[0];
        unsigned int mHeight = imSize[1];
        unsigned int mDepth = imSize[2];
        unsigned int mComp = img->GetNumberOfComponentsPerPixel();


        if ( (mWidth != rawImage.width()) || (mHeight != rawImage.height()) || (mDepth != rawImage.depth()) )
            qFatal("Feature image size differs from raw image: %s", featFName.c_str());

        // create as many integral images as the number of channels
        std::vector<IntegralImage<IntegralImageType> *> thisIIs;
        for (unsigned q=0; q < mComp; q++)
            thisIIs.push_back( new IntegralImage<IntegralImageType>() );

        for (unsigned q=0; q < mComp; q++)
        {
            Matrix3D<float> auxImg;
            auxImg.reallocSizeLike( rawImage );

            for (unsigned pix=0; pix < rawImage.numElem(); pix++)
            {
                ItkVectorImageType::IndexType index;

                unsigned x,y,z;
                rawImage.idxToCoord(pix, x, y, z);

                index[0] = x;
                index[1] = y;
                index[2] = z;

                const ItkVectorImageType::PixelType &pixData = img->GetPixel(index);
                auxImg.data()[pix] = pixData[q];
            }

            // now compute II
            IntegralImage<IntegralImageType> *ii = thisIIs.at(q);

            ii->compute( auxImg );
        }

        // now we can append it
        #pragma omp ordered
        {
            // resize orig
            const unsigned prevN = IIs.size();

            for (unsigned q=0; q < mComp; q++)
            {
                IIs.push_back( thisIIs.at(q) );
                qPrint("-- Appended %s to idx [%d,%d]", featFName.c_str(), prevN, prevN + mComp - 1);

#if APPLY_PATCH_VARNORM
                // this always stays the same, except squared for structure tensor
                combo.svoxWindowInvStd.push_back( combo.svoxWindowInvStd[0] );

                // if we have pixel values, keep mean, otherwise set it to zero
                if ( featFName.find( "gauss" ) != std::string::npos )
                {
                    qDebug("Adding intensity");
                    combo.svoxWindowMean.push_back( combo.svoxWindowMean[0] );
                }
                else
                {
                    qDebug("Adding non-intensity");
                    combo.svoxWindowMean.push_back( std::vector<float>() );
                    combo.svoxWindowMean.back().resize( combo.svoxWindowMean[0].size(), 0.0 );

                    // structure tensor
                    if ( featFName.find( "tens" ) != std::string::npos )
                    {
                        qDebug("Specializing for tensor");
                        for (unsigned r=0; r < combo.svoxWindowInvStd[0].size(); r++)
                            combo.svoxWindowInvStd.back()[r] = combo.svoxWindowInvStd.back()[r] * combo.svoxWindowInvStd.back()[r]; // square!
                    }
                }
#endif
            }
        }
    }
}

// if dontUseHistograms == true => only mean value is used in histogram
void computeSupervoxelCombo( const SetConfigData &cfgData,
                             int svStep, int svCubeness,
                             SVCombo &combo)
{
    qPrint("Computing supervoxel combo");
    SuperVoxeler<PixelType> &SVox = combo.SVox;
    std::vector<UIntPoint3D> &svCentroids = combo.svCentroids;

    Matrix3D<PixelType>  &rawImage = combo.rawImage;

    const std::string &volumeFileName = cfgData.rawVolume;

    qPrint("Loading image file");
    //qDebug("Loading image %s", cfgData.rawVolume.c_str());
    if (!rawImage.load( cfgData.rawVolume ))
        qFatal("Error loading volume: %s", cfgData.rawVolume.c_str());

    if (!LoadSVox( volumeFileName, svStep, svCubeness, combo.SVox ))
    {
        qPrint("Computing supervoxels %d-%d", svStep, svCubeness);


        const int svZStep = std::max( (int) ( svStep / cfgData.zAnisotropyFactor), 1 );
        qPrint("SVox Step / zStep: %d, %d", svStep, svZStep);

        combo.SVox.apply( rawImage, svStep, svCubeness, svZStep );

        // save it
        qPrint("Saving supervoxels");
        SaveSVox( volumeFileName, svStep, svCubeness, SVox );
    }

    qPrint("Computing supervoxel centroids");
    /** Compute centroids **/
    svCentroids.resize( SVox.numLabels() );
    for (unsigned int i=0; i < SVox.numLabels(); i++)
    {
        UIntPoint3D pt;
        computeSVCenter( SVox.voxelToPixel()[i], &pt );
        svCentroids[i] = pt;
    }

    /** -------------------------------------- **/
    qPrint("Computing integral image");
    combo.pixIntegralImages.clear();
    combo.pixIntegralImages.push_back( new IntegralImage<IntegralImageType>() );
    combo.pixIntegralImages.back()->compute( combo.rawImage );

#if APPLY_PATCH_VARNORM
    computeSvoxWindowStatistics( combo, combo.pixIntegralImages.size() - 1, combo.rawImage );
#endif

#if APPLY_PATCH_VARNORM
    // same as before
    combo.svoxWindowMean.push_back( combo.svoxWindowMean[0] );
    combo.svoxWindowInvStd.push_back( combo.svoxWindowInvStd[0] );
#endif

    /** Get orientation **/
#if !USE_ALL_EIGVEC_FOR_ROTMATRICES
    combo.svOrient.resize( SVox.numLabels() );
    qPrint("Reading orientation image");
    if (true)
    {
        typedef itk::VectorImage<float, 3>  ItkVectorImageType;
        itk::ImageFileReader<ItkVectorImageType>::Pointer reader = itk::ImageFileReader<ItkVectorImageType>::New();
        reader->SetFileName( cfgData.orientEstimate );
        reader->Update();

        ItkVectorImageType::Pointer img = reader->GetOutput();


        ItkVectorImageType::SizeType imSize = img->GetLargestPossibleRegion().GetSize();
        unsigned int mWidth = imSize[0];
        unsigned int mHeight = imSize[1];
        unsigned int mDepth = imSize[2];
        unsigned int mComp = img->GetNumberOfComponentsPerPixel();

        if ( (mWidth != rawImage.width()) || (mHeight != rawImage.height()) || (mDepth != rawImage.depth()) )
            qFatal("Orientation image size differs from raw image");

        if (mComp != 3)
            qFatal("Vector components must be 3 (is %d)", (int)mComp);

        for (unsigned int i=0; i < svCentroids.size(); i++)
        {
            ItkVectorImageType::IndexType index;
            index[0] = svCentroids[i].x;
            index[1] = svCentroids[i].y;
            index[2] = svCentroids[i].z;

            const ItkVectorImageType::PixelType &pixData = img->GetPixel(index);

            combo.svOrient[i].x = pixData[0];
            combo.svOrient[i].y = pixData[1];
            combo.svOrient[i].z = pixData[2];
        }

        //qDebug("Pix: %f %f %f", svOrient[10].x, svOrient[10].y, svOrient[10].z);
    }
    else
    {
        for (unsigned int i=0; i < svCentroids.size(); i++)
        {
            combo.svOrient[i].x = rand() * 1.0 / RAND_MAX;
            combo.svOrient[i].y = rand() * 1.0 / RAND_MAX;
            combo.svOrient[i].z = rand() * 1.0 / RAND_MAX;
        }
    }

    qPrint("Computing rotation matrices");
    computeRotationMatrices( combo.svOrient, combo.rotMatrices );
#else
    combo.rotMatrices.resize( SVox.numLabels() );
    qPrint("Reading orientation image");

    if (true)
    {
        typedef itk::VectorImage<float, 3>  ItkVectorImageType;
        itk::ImageFileReader<ItkVectorImageType>::Pointer reader = itk::ImageFileReader<ItkVectorImageType>::New();
        reader->SetFileName( cfgData.orientEstimate );
        reader->Update();

        ItkVectorImageType::Pointer img = reader->GetOutput();


        ItkVectorImageType::SizeType imSize = img->GetLargestPossibleRegion().GetSize();
        unsigned int mWidth = imSize[0];
        unsigned int mHeight = imSize[1];
        unsigned int mDepth = imSize[2];
        unsigned int mComp = img->GetNumberOfComponentsPerPixel();

        if ( (mWidth != rawImage.width()) || (mHeight != rawImage.height()) || (mDepth != rawImage.depth()) )
            qFatal("Orientation image size differs from raw image");

        if (mComp != 9)
            qFatal("Vector components must be 9 / all eigenvector data (but is %d)", (int)mComp);

        for (unsigned int i=0; i < combo.rotMatrices.size(); i++)
        {
            ItkVectorImageType::IndexType index;
            index[0] = svCentroids[i].x;
            index[1] = svCentroids[i].y;
            index[2] = svCentroids[i].z;

            const ItkVectorImageType::PixelType &pixData = img->GetPixel(index);

            // first eigvec, this will be computed from a vector product
            //combo.rotMatrices[i].coeffRef( 0, 0 ) = pixData[0];
            //combo.rotMatrices[i].coeffRef( 1, 0 ) = pixData[1];
            //combo.rotMatrices[i].coeffRef( 2, 0 ) = pixData[2];

            // 2nd eigvec
            combo.rotMatrices[i].coeffRef( 0, 1 ) = pixData[3];
            combo.rotMatrices[i].coeffRef( 1, 1 ) = pixData[4];
            combo.rotMatrices[i].coeffRef( 2, 1 ) = pixData[5];

            // 3rd eigvec
            combo.rotMatrices[i].coeffRef( 0, 2 ) = pixData[6];
            combo.rotMatrices[i].coeffRef( 1, 2 ) = pixData[7];
            combo.rotMatrices[i].coeffRef( 2, 2 ) = pixData[8];

            // ex = ey .cross. ez (right-handed coord system)
            combo.rotMatrices[i].col(0) = combo.rotMatrices[i].col(1).cross( combo.rotMatrices[i].col(2) );


#if IGNORE_ORIENT_ESTIMATES
            combo.rotMatrices[i].setIdentity();
#endif
        }

        //qDebug("Pix: %f %f %f", svOrient[10].x, svOrient[10].y, svOrient[10].z);
    }
#endif

    // add extra features
    const std::vector< std::string > &extraFeaturesFNames = cfgData.otherFeatures;
    //#pragma omp parallel for num_threads(3)  // just to speed up loading from HD => PROBLEM IS ORDER!
    {
        qPrint("Appending features");
        appendFeatures( extraFeaturesFNames,
                        rawImage,
                        combo.pixIntegralImages,
                        combo );
    }
#if APPLY_PATCH_VARNORM
    qPrint("Num wnd mean/invStd: %d %d", (int)combo.svoxWindowMean.size(), (int)combo.svoxWindowInvStd.size());
#endif

#if USE_SPATIAL_BOOSTING
    // add II, blank for now
    combo.pixIntegralImages.push_back( new IntegralImage<IntegralImageType>() );
    combo.pixIntegralImages.back()->initializeToZero( rawImage );
#endif
}

template<typename AdaBoostType>
void saveTrainingWeights( const ConfigData &cfgData, const AdaBoostType &adaboost, const SVCombo &combo,
                          const std::vector<unsigned int>  &appendedPositivesAsNegativesIdx)
{
    qPrint("Saving weights");
    Matrix3D<float>     weightImg;
    weightImg.reallocSizeLike( combo.rawImage );

    const typename AdaBoostType::WeightsType        &weights = adaboost.getSampleWeights();
    const typename AdaBoostType::SampleIdxVector    &sampleIdxs = adaboost.getSampleIdxs();
    const typename AdaBoostType::SampleClassVector  &classes = adaboost.getSampleClasses();

    const unsigned int totalSVox = combo.SVox.numLabels();

    //negative samples
    {
        weightImg.fill(0);

        unsigned int appendedIdxsStartAt = 0;
        if (appendedPositivesAsNegativesIdx.size() > 0)
        {
            typename std::vector<unsigned int>::const_iterator it = std::find( sampleIdxs.begin(), sampleIdxs.end(), appendedPositivesAsNegativesIdx[0] );
            appendedIdxsStartAt = it - sampleIdxs.begin();
        }

        for (unsigned i=0; i < sampleIdxs.size(); i++)
        {
            if ( classes[i] == 1 )  continue;

            unsigned idx = i;
            if (idx >= appendedIdxsStartAt) // then
                idx -= appendedIdxsStartAt;   // TRICK BUT DEPENDS ON ORDERING, ASSUMING TOO MUCH!

            const PixelInfoList &pixList = combo.SVox.voxelToPixel().at( sampleIdxs[idx] );
            const float val = weights[i];
            for (unsigned int p=0; p < pixList.size(); p++)
                weightImg.data()[ pixList[p].index ] = val;
        }

        weightImg.save( cfgData.saveTrainingWeightsNegFileName );
    }

    //positive samples
    {
        weightImg.fill(0);

        for (unsigned i=0; i < sampleIdxs.size(); i++)
        {
            if ( classes[i] != 1 )  continue;

            const PixelInfoList &pixList = combo.SVox.voxelToPixel().at(sampleIdxs[i]);
            const float val = weights[i];
            for (unsigned int p=0; p < pixList.size(); p++)
                weightImg.data()[ pixList[p].index ] = val;
        }

        weightImg.save( cfgData.saveTrainingWeightsPosFileName );
    }
}

int main(int argc, char **argv)
{
    qPrint("Welcome to CCBOOST");

    srand(1234);

    qDebug("Integral image type: %s", typeToString<IntegralImageType>());
    if ( typeid( IntegralImageType ) != typeid(double) )
    {
        qPrint("Warning: 'double' is the recommended type for IntegralImageType, "
               "using 'float' may introduce undesirable artifacts");
    }

#if 0
    IntegralImage<IntegralImageType>    ii;
    Matrix3D<float> test;
    test.realloc(1000,1000,700);
    for (unsigned i=0; i < test.numElem(); i++)
        test.data()[i] = rand() * 1.0 / RAND_MAX;
    //test.data()[i] = 10;

    ii.compute(test);

    unsigned x1 = 800;
    unsigned x2 = x1+10;

    unsigned y1 = 800;
    unsigned y2 = y1+10;

    unsigned z1 = 600;
    unsigned z2 = z1+10;

    qDebug("II:   %f", ii.volumeSum( x1, x2, y1, y2, z1, z2 ) );
    //qDebug("II c: %f", ii.centeredSumNormalized( x1, y1, z1, 7 ) );
    qDebug("pix:  %d", test(x1,y1,z1));

    unsigned t1,t2,t3;
    test.idxToCoord( test(x1,y1,z1), t1, t2, t3 );
    qDebug("Coord: %d %d %d\n", t1,t2,t3);

    double sum = 0;

    qDebug("Offset: %d\n", test.coordToIdx(x1,y1,z1) );

    for (unsigned x=x1; x <= x2; x++)
        for (unsigned y=y1; y <= y2; y++)
            for (unsigned z=z1; z <= z2; z++)
                sum += test(x,y,z);
    qDebug("Normal: %f", sum);

    return 0;
#endif

    if (argc < 2)
        qFatal("Wrong arguments, usage is:\n$ %s <configFile.cfg> <optional args>", argv[0]);

    std::string extraConfigLines = "";

    if (argc >= 3)
        extraConfigLines = std::string(argv[2]);

    ConfigData  cfgData;
    try
    {
        cfgData.open( argv[1], true, extraConfigLines );
    }
    catch(const libconfig::FileIOException &fioex)
    {
        std::cout << "I/O error while reading file." << std::endl;
        return(EXIT_FAILURE);
    }
    catch(const libconfig::ParseException &pex)
    {
        std::cout << "Configuration Parse error: " << pex.getFile() << ":" << pex.getLine()
                  << " - " << pex.getError() << std::endl;
        return(EXIT_FAILURE);
    }
    catch(const libconfig::SettingNotFoundException &pex)
    {

        std::cout << "Setting not found: " << pex.getPath() << std::endl;
        return(EXIT_FAILURE);
    }

    //cfgData.printInfo();

    bool useSingleSynapse = cfgData.useSynapses.size() != 0;

    if (useSingleSynapse)
    {
        for (unsigned i=0; i < cfgData.useSynapses.size(); i++)
            qDebug("Synapse: %d", cfgData.useSynapses[i]);
    }

    // compute supervoxels

    const int svStep = cfgData.svoxSeed;
    const int svCubeness = cfgData.svoxCubeness;

    typedef AdaBoost<HistogramMeanThreshold> AdaBoostType;
    AdaBoostType adaboost;

    SigmoidFitterPlatt<double> sigmoidPlatt;

    // histogram options
    /*HistogramOpts<PixelType>    hOpts;
    hOpts.begin = 0;
    hOpts.end = 255;
    hOpts.nbins = 20;*/

    if (!cfgData.savedStumpsPath.empty())
    {
        // then load stumps
        adaboost.loadStumps( cfgData.savedStumpsPath, sigmoidPlatt );

        // see numStumps
        if (cfgData.numStumps != adaboost.numTrainedStumps())
        {
            qPrint("Limiting stumps to the first %d", (int)cfgData.numStumps);
            adaboost.cropStumpsTo( cfgData.numStumps );
            qPrint("Got: %d stumps", (int) adaboost.numTrainedStumps());
        }
    }

    /** --- TRAIN LOOP, only if stumps not provided and the user doesn't want them ***/
    //std::cout << "STUMPS PATH VAL: " << cfgData.savedStumpsPath << std::endl;
    //std::cout << "STUMPS TESTING VAL: " << cfgData.usedSavedStumpsPathOnlyForTesting << std::endl;
    //exit(-1);

    if ( !cfgData.usedSavedStumpsPathOnlyForTesting )
    {
        qPrint("STARTING TRAINING!");
        SVCombo trainCombo;
        computeSupervoxelCombo( cfgData.train, svStep, svCubeness, trainCombo );

#if 0
        {
            unsigned x1 = 220;
            unsigned x2 = x1  + 20;

            unsigned y1 = 98;
            unsigned y2 = y1 + 12;

            unsigned z1 = 130;
            unsigned z2 = z1 + 7;

            qDebug("Num IIs: %d", (int) trainCombo.pixIntegralImages.size());
            IntegralImage<IntegralImageType> &ii = *trainCombo.pixIntegralImages[0];
            qDebug("II:   %f", ii.volumeSum( x1, x2, y1, y2, z1, z2 ) );
            qDebug("II+1: %f", ii.volumeSum( x1, x2, y1, y2, z1, z2 ) );
            //qDebug("II c: %f", ii.centeredSumNormalized( x1, y1, z1, 0 ) );
            qDebug("pix:  %d", trainCombo.rawImage(x1,y1,z1));

            double sum = 0;
            for (unsigned x=x1; x <= x2; x++)
                for (unsigned y=y1; y <= y2; y++)
                    for (unsigned z=z1; z <= z2; z++)
                        sum += trainCombo.rawImage(x,y,z);
            qDebug("Normal: %f", sum);
        }
#endif

        // create some references to ease notation
        SuperVoxeler<PixelType> &SVox = trainCombo.SVox;
        std::vector<UIntPoint3D> &svCentroids = trainCombo.svCentroids;
        std::vector<Eigen::Matrix3f> &rotMatrices = trainCombo.rotMatrices;

        qPrint("Loading GT");
        const PixelType posLabel = 255;
        const PixelType negLabel = 0;

        Matrix3D<PixelType> gtImage;
        if (!gtImage.load( cfgData.train.groundTruth ))
            qFatal("Could not load GT volume");

        if ( !gtImage.isSizeLike(trainCombo.rawImage)  )
            qFatal("GT and raw image size don't match");

        // do we have to erode the image?
        if ( cfgData.train.groundTruthErodeRadius > 0 )
        {
            qDebug("Eroding GT with ball size %d", (int)cfgData.train.groundTruthErodeRadius);

            typedef itk::BinaryBallStructuringElement<PixelType, 3> StructuringElementType;
            StructuringElementType structuringElement;
            structuringElement.SetRadius( cfgData.train.groundTruthErodeRadius );
            structuringElement.CreateStructuringElement();

            typedef itk::BinaryErodeImageFilter< itk::Image<PixelType,3>, itk::Image<PixelType,3>, StructuringElementType>
            BinaryErodeImageFilterType;

            BinaryErodeImageFilterType::Pointer erodeFilter = BinaryErodeImageFilterType::New();
            erodeFilter->SetInput( gtImage.asItkImage() );
            erodeFilter->SetKernel( structuringElement );
            erodeFilter->SetErodeValue( posLabel );

            erodeFilter->Update();

            gtImage.copyFrom( erodeFilter->GetOutput() );
        }

        // find supervoxels with pos / neg
        std::vector<unsigned int>   posSVIdx, negSVIdx;
        {
            // add unique supervoxel idxs + count
            std::map<unsigned int, unsigned int>  negIdxs;
            const PixelType *gtData = gtImage.data();
            const unsigned int *svoxData = SVox.pixelToVoxel().data();

            for (unsigned int i=0; i < gtImage.numElem(); i++)
            {
                /*if ( gtData[i] == posLabel )
                    posIdxs[ svoxData[i] ]++;*/

                //qDebug("Here");
                if ( gtData[i] == negLabel )
                    //std::cout << svoxData[i] << '-' << negIdxs[ svoxData[i] ]++ << std::endl;
                    negIdxs[ svoxData[i] ]++;
            }

            // now go through each element in the map and check votings
            const double minProp = 0.8;
            for ( std::map<unsigned int, unsigned int>::iterator it=negIdxs.begin() ; it != negIdxs.end(); it++ )
            {
                unsigned int idx = it->first;
                unsigned int count = it->second;

                if ( count * 1.0 / SVox.voxelToPixel()[idx].size() >= minProp )
                    negSVIdx.push_back( idx );
            }

            /** for positives, we run connected comp analysis on GT image **/
            std::vector< std::vector<unsigned int> >    posIdxBags;
            std::vector< std::vector<unsigned int> >    posSVBags;
            bool foundPolarityError = false;
            {
                const unsigned int minPosRegionSize = cfgData.minPosRegionSize;

                ConnectedComponents<PixelType> CComp;
                CComp.process( gtImage, posLabel, posLabel );

                CComp.findRegionsBiggerThan( minPosRegionSize, posIdxBags );

                // before starting, convert [px,py,pz] to the supervoxel it belongs to
                //  to make things faster
                std::vector<unsigned>  polarityCfgSVoxIndex( cfgData.train.ccGTOrient.size() );
                for (unsigned i=0; i < polarityCfgSVoxIndex.size(); i++)
                {
                    unsigned svoxIdx = trainCombo.SVox.pixelToVoxel() ( cfgData.train.ccGTOrient[i].px, cfgData.train.ccGTOrient[i].py, cfgData.train.ccGTOrient[i].pz );
                    qDebug("Polarity info for %d %d %d found. SVox %d", cfgData.train.ccGTOrient[i].px, cfgData.train.ccGTOrient[i].py, cfgData.train.ccGTOrient[i].pz, svoxIdx);
                    polarityCfgSVoxIndex[i]= svoxIdx;
                }


                const double minProp = 0.9;
                for (unsigned int i=0; i < posIdxBags.size(); i++)
                {
                    // only process if it is the one we want
                    if ( useSingleSynapse )
                    {
                        if ( std::find( cfgData.useSynapses.begin(), cfgData.useSynapses.end(), i+1 ) == cfgData.useSynapses.end() )
                        {
                            qDebug("Skipping synapse %d", i+1);
                            continue;
                        }
                    }

                    std::map<unsigned int, unsigned int> mmap;

                    for (unsigned int j=0; j < posIdxBags[i].size(); j++)
                        mmap[ svoxData[ posIdxBags[i][j] ] ]++;

                    // check proportions and add to list
                    std::vector<unsigned int> SVList;
                    for ( std::map<unsigned int, unsigned int>::iterator it=mmap.begin() ; it != mmap.end(); it++ )
                    {
                        unsigned int idx = it->first;
                        unsigned int count = it->second;

                        if ( count * 1.0 / SVox.voxelToPixel()[idx].size() >= minProp )
                            SVList.push_back( idx );
                    }

                    if (SVList.size() == 0)
                        continue;

#if USE_POLARITY
                    // find to which polarity info in the cfg file this region belongs to
                    Eigen::Vector3f polarityVec;
                    {
                        std::vector<unsigned> foundOnes;
                        for (unsigned q=0; q < SVList.size(); q++)
                        {
                            for (unsigned w=0; w < polarityCfgSVoxIndex.size(); w++)
                            {
                                if ( SVList[q] == polarityCfgSVoxIndex[w] )
                                    foundOnes.push_back( w );
                            }
                        }

                        // error check
                        if ( foundOnes.size() != 1 )
                        {
                            foundPolarityError = true;
                            qDebug("Error assigning polarity with config: %d, size %d, for CC at %d %d %d. SV %d ",
                                   (int) foundOnes.size(),
                                   (int) posIdxBags[i].size(),
                                   (int)svCentroids[SVList[ SVList.size() / 2 ]].x,
                                   (int) svCentroids[SVList[SVList.size() / 2]].y, (int)svCentroids[SVList[SVList.size() / 2]].z,
                                   SVList[SVList.size()/2] );

                            if (foundOnes.size() != 0)
                            {
                                qDebug("Found ones:\n");
                                for (unsigned r=0; r < foundOnes.size(); r++)
                                {
                                    const unsigned e = foundOnes[r];
                                    qDebug("\t%d %d %d\n", cfgData.train.ccGTOrient[e].px, cfgData.train.ccGTOrient[e].py, cfgData.train.ccGTOrient[e].pz );
                                }
                            }
                            continue;
                        }

                        polarityVec << cfgData.train.ccGTOrient[foundOnes[0]].vx, cfgData.train.ccGTOrient[foundOnes[0]].vy, cfgData.train.ccGTOrient[foundOnes[0]].vz;
                    }

                    qDebug("Num svoxels: %d", (int) SVList.size());
                    unsigned bfSVListIdx = 0;
                    Eigen::Vector3f bfNormal;
                    float bfCost = 0;
                    {
                        bruteForceNormalSearch( rotMatrices, SVList, &bfNormal, &bfSVListIdx, &bfCost );
                        qDebug("BF cost: %f\tNormal: %f %f %f", bfCost, bfNormal(0), bfNormal(1), bfNormal(2));
                    }

                    // see if we should invert bfNormal
                    if ( (bfNormal.array() * polarityVec.array()).sum() < 0 )
                        bfNormal = -bfNormal;

                    // create matrix to know which ones to revert
                    Eigen::MatrixXf vecs( 3, SVList.size() );
                    for (unsigned q=0; q < SVList.size(); q++)
                    {
                        vecs.col(q) = rotMatrices[ SVList[q] ].col(2);
                    }

                    // compute dot prods
                    Eigen::VectorXf dotP = bfNormal.transpose() * vecs;

                    // set with same orient by inverting the ones with the 'wrong'  one
                    for (unsigned q=0; q < SVList.size(); q++)
                    {
                        if ( dotP(q) < 0 )
                        {
                            rotMatrices[ SVList[q] ].col(0) = -trainCombo.rotMatrices[ SVList[q] ].col(0);
                            rotMatrices[ SVList[q] ].col(2) = -trainCombo.rotMatrices[ SVList[q] ].col(2);
                        }
                    }
#else
                    const unsigned bfSVListIdx = SVList.size() / 2;
                    const double bfCost = 0.0;
#endif

                    // show location as indicator
                    {
                        qDebug("GT Bag %d: %.2f %.2f %.2f (cost %.2f, size %d)", i+1,
                               rotMatrices[ SVList[bfSVListIdx] ].coeff(0,2),
                               rotMatrices[ SVList[bfSVListIdx] ].coeff(1,2),
                               rotMatrices[ SVList[bfSVListIdx] ].coeff(2,2),
                               bfCost, (int) SVList.size() );


                        qDebug("   at %d %d %d", (int)svCentroids[SVList[bfSVListIdx]].x,
                               (int) svCentroids[SVList[bfSVListIdx]].y, (int)svCentroids[SVList[bfSVListIdx]].z );
                    }

                    posSVBags.push_back( SVList );

                    // and add to the pix list
                    for (unsigned q=0; q < SVList.size(); q++)
                        posSVIdx.push_back( SVList[q] );
                }

                if (foundPolarityError)
                    qFatal("Breaking because of polarity error, see above.");
            }
        }

        // remove the ones on the border
        {
            std::vector<unsigned int> newPosSV, newNegSV;

            const unsigned int minDist = cfgData.borderMinDist;

            // scale for z
            const unsigned int minDistZ = cfgData.borderMinDist / cfgData.train.zAnisotropyFactor;

            const unsigned int maxX = gtImage.width() - minDist;
            const unsigned int maxY = gtImage.height() - minDist;
            const unsigned int maxZ = gtImage.depth() - minDistZ;

            for (unsigned i=0; i < posSVIdx.size(); i++)
            {
                if (svCentroids[ posSVIdx[i] ].x < minDist)	continue;
                if (svCentroids[ posSVIdx[i] ].y < minDist)	continue;
                if (svCentroids[ posSVIdx[i] ].z < minDistZ)	continue;

                if (svCentroids[ posSVIdx[i] ].x > maxX)	continue;
                if (svCentroids[ posSVIdx[i] ].y > maxY)	continue;
                if (svCentroids[ posSVIdx[i] ].z > maxZ)	continue;

                newPosSV.push_back( posSVIdx[i] );
            }

            for (unsigned i=0; i < negSVIdx.size(); i++)
            {
                if (svCentroids[ negSVIdx[i] ].x < minDist)	continue;
                if (svCentroids[ negSVIdx[i] ].y < minDist)	continue;
                if (svCentroids[ negSVIdx[i] ].z < minDistZ)	continue;

                if (svCentroids[ negSVIdx[i] ].x > maxX)	continue;
                if (svCentroids[ negSVIdx[i] ].y > maxY)	continue;
                if (svCentroids[ negSVIdx[i] ].z > maxZ)	continue;

                newNegSV.push_back( negSVIdx[i] );
            }

            posSVIdx = newPosSV;
            negSVIdx = newNegSV;
        }

        // resample negSVIdx according to proportion
        {
            double p = cfgData.getKeyValue<double>( std::string("NegSampleProportion") );

            if (p != 0.0)
            {
                qPrint("Resampling negative class!!");
                unsigned nn = p * posSVIdx.size() + 0.5;

                std::vector<unsigned int> newIdxs;
                sampleWithoutReplacement( nn, negSVIdx.size(), &newIdxs );

                std::vector<unsigned int> oldSVNeg = negSVIdx;
                negSVIdx.clear();

                for (unsigned i=0; i < newIdxs.size(); i++)
                    negSVIdx.push_back( oldSVNeg[ newIdxs[i] ] );
            }
            else
                qPrint("Using ALL negative-labeled supervoxels");
        }

        // resample posSVIdx according to proportion
        {
            unsigned p = cfgData.getKeyValue<int>( std::string("PosSkipFactor") );

            if (p != 0)
            {
                qPrint("Resampling POSITIVE class!!");
                unsigned nn = p * posSVIdx.size() + 0.5;

                std::vector<unsigned> newIdxs;

                for (unsigned q=0; q < posSVIdx.size(); q += p)
                    newIdxs.push_back( posSVIdx[q] );

                posSVIdx = newIdxs;
            }
            else
                qPrint("Using ALL negative-labeled supervoxels");
        }


        // take out 1 out of 3
        if(false)
        {
            qDebug("------> WARNING: sampling 1 out of 3!");
            std::random_shuffle(posSVIdx.begin(), posSVIdx.end());

            posSVIdx.erase( posSVIdx.begin() + 2*(posSVIdx.size())/3, posSVIdx.end() );


            std::random_shuffle(negSVIdx.begin(), negSVIdx.end());

            negSVIdx.erase( negSVIdx.begin() + (2*negSVIdx.size())/3, negSVIdx.end() );
        }

        qPrint("GT Pos SV: %d", (int)posSVIdx.size());
        qPrint("GT Neg SV: %d", (int)negSVIdx.size());

        if ( (posSVIdx.size() == 0) || (negSVIdx.size() == 0) )
            qFatal("Not enough ground truth!");

        // shuffle all orientations from negatives
        for (unsigned q=0; q < negSVIdx.size(); q += 2)
        {
            unsigned idx = negSVIdx[q];
            rotMatrices[idx].col(0) = -rotMatrices[idx].col(0);
            rotMatrices[idx].col(2) = -rotMatrices[idx].col(2);
        }


        const bool appendPositivesAsNegatives = cfgData.appendArtificialNegatives;
        std::vector<unsigned int>  appendedPositivesAsNegativesIdx;
        if (appendPositivesAsNegatives)
        {
            qPrint("Appending positives as negatives");
            const unsigned int kMax = 1;

            const unsigned int prevRows = trainCombo.SVox.numLabels();
            /*
            histMatrix.conservativeResize( histMatrix.rows() + kMax * posSVIdx.size(), histMatrix.cols() );*/

            unsigned int ii = prevRows;
            for (unsigned K=0; K < kMax; K++)
            {
                for (unsigned i=0; i < posSVIdx.size(); i++)
                {
                    unsigned prevSVIdx = posSVIdx[i];

                    unsigned newSVIdx = ii++;

                    negSVIdx.push_back( newSVIdx );
                    svCentroids.push_back( svCentroids[ prevSVIdx ] );

                    rotMatrices.push_back( -rotMatrices[ prevSVIdx ] );

                    //histMatrix.row( newSVIdx ) = histMatrix.row( prevSVIdx );

                    appendedPositivesAsNegativesIdx.push_back( newSVIdx );

                    // also duplicate std dev / mean
#if APPLY_PATCH_VARNORM
                    for (unsigned q=0; q < trainCombo.svoxWindowMean.size(); q++)
                    {
                        trainCombo.svoxWindowMean[q].push_back( trainCombo.svoxWindowMean[q][prevSVIdx] );
                        trainCombo.svoxWindowInvStd[q].push_back( trainCombo.svoxWindowInvStd[q][prevSVIdx] );
                    }
#endif
                }
            }
        }
        else
            qPrint("WARNING: not adding artificial negatives!");

        // debugging stuff
        if ( appendPositivesAsNegatives && false )
        {
            qPrint("Running negatives/positives test");

            for (unsigned w=0; w < 10; w++)
            {
                const unsigned posIdx = posSVIdx[w];
                const unsigned negIdx = appendedPositivesAsNegativesIdx[w];
                qDebug("CenterP: %d %d %d", svCentroids[posIdx].x, svCentroids[posIdx].y, svCentroids[posIdx].z );
                qDebug("CenterN: %d %d %d", svCentroids[negIdx].x, svCentroids[negIdx].y, svCentroids[negIdx].z );
                qDebug("OrientP: %.1f %.1f %.1f", rotMatrices[posIdx].coeff(0,2), rotMatrices[posIdx].coeff(1,2), rotMatrices[posIdx].coeff(2,2));
                qDebug("OrientN: %.1f %.1f %.1f", rotMatrices[negIdx].coeff(0,2), rotMatrices[negIdx].coeff(1,2), rotMatrices[negIdx].coeff(2,2));
                qDebug("MatrixP:");
                std::cout << trainCombo.rotMatrices[posIdx] << std::endl;
                qDebug("MatrixN:");
                std::cout << trainCombo.rotMatrices[negIdx] << std::endl;
                qDebug("-----------------------------------------");
            }

            qDebug("------- End Negative / positives test");
        }


        qPrint("Integral images: %d", (int) trainCombo.pixIntegralImages.size() );


        // concatenate pos and neg
        std::vector<unsigned int>  samplesIdx(posSVIdx.size() + negSVIdx.size());
        std::vector<unsigned int>  samplesClass( samplesIdx.size() );
        for (unsigned int i=0; i < posSVIdx.size(); i++)
        {
            samplesIdx[i] = posSVIdx[i];
            samplesClass[i] = 1;
        }

        for (unsigned int i=0; i < negSVIdx.size(); i++)
        {
            samplesIdx[posSVIdx.size() + i] = negSVIdx[i];
            samplesClass[posSVIdx.size() + i] = 0;
        }


#if 0
        for (unsigned int i=0; i < samplesClass.size(); i++)
        {
            histMatrix( samplesIdx[i], 1) = (samplesClass[i] == 1)?10:20;
        }
        histMatrix(samplesIdx[20], 1) = 200;
        histMatrix(samplesIdx[21], 1) = 3;
        histMatrix(samplesIdx[22], 1) = 4;
        histMatrix(samplesIdx[23], 1) = 5;
#endif

        HistogramMeanThresholdData params( svCentroids, trainCombo.rotMatrices, trainCombo.pixIntegralImages, SVox.pixelToVoxel(),
                                           cfgData.train.zAnisotropyFactor
#if APPLY_PATCH_VARNORM
                                           , trainCombo.svoxWindowInvStd, trainCombo.svoxWindowMean );
#else
                                         );
#endif

        params.numWLToExplore = cfgData.numWLToExplorePerIter;
        adaboost.init( params, samplesIdx, samplesClass );

        adaboost.setReweightingSamplingRatio( cfgData.reweightingSamplingRatio );
        adaboost.setThresholdSearchOnWholeData( cfgData.thresholdSearchOnWholeData);

        // to tell the adaboost class which samples are artifneg
        std::vector<unsigned char> isArtifNeg( samplesIdx.size(), 0 );
        for (unsigned i=0; i < appendedPositivesAsNegativesIdx.size(); i++)
        {
            std::vector<unsigned int>::iterator it = std::find( samplesIdx.begin(), samplesIdx.end(), appendedPositivesAsNegativesIdx[i] );

            if ( it == samplesIdx.end() )
                qFatal("Not found, something wrong.");

            isArtifNeg[(it - samplesIdx.begin())] = 1;
        }

        adaboost.setArtifNegIdxs( isArtifNeg );

        if ( appendPositivesAsNegatives && cfgData.treatArtificialNegativesAsPositivesInResampling )   // then we have to modify the minority idxs
        {
            qPrint("Modifying minority idxs for negative samples");
            std::vector<unsigned char> minClassIdx( samplesIdx.size(), 0 );
            for (unsigned i=0; i < posSVIdx.size(); i++)
            {
                std::vector<unsigned int>::iterator it = std::find( samplesIdx.begin(), samplesIdx.end(), posSVIdx[i] );

                if ( it == samplesIdx.end() )
                    qFatal("Not found, something wrong.");

                minClassIdx[(it - samplesIdx.begin())] = 1;
            }

            for (unsigned i=0; i < appendedPositivesAsNegativesIdx.size(); i++)
            {
                std::vector<unsigned int>::iterator it = std::find( samplesIdx.begin(), samplesIdx.end(), appendedPositivesAsNegativesIdx[i] );

                if ( it == samplesIdx.end() )
                    qFatal("Not found, something wrong.");

                minClassIdx[(it - samplesIdx.begin())] = 1;
            }

            adaboost.setMinorityClassIdx( minClassIdx );
        }

        adaboost.setBackupLearnedStumpsFileName( cfgData.outFileName + "-backupstumps.cfg" );

        // only learn if stumps were not provided
        if (cfgData.savedStumpsPath.empty() || cfgData.relearnThresholds)
        {
            qPrint("Adaboost learn");
            adaboost.learn( cfgData.numStumps, trainCombo, cfgData.relearnThresholds );
        }
        else
        {
            qPrint("Not learning because stumps were provided");
        }

        adaboost.printSplitInformation();

        // do we have to write down the pos/neg training samples?
        if ( cfgData.keyExists("extra.saveTrainingFeatures") )
        {
            bool doIt;
            cfgData.getKeyValue("extra.saveTrainingFeatures", doIt);

            if (doIt)
            {
                qPrint("Saving features (should not be here)");

                Eigen::ArrayXXf feats;

                adaboost.exportFeatures( posSVIdx, feats );
                writeMatrix( cfgData.outFileName + "-trainPosFeats.bin", feats );

                adaboost.exportFeatures( negSVIdx, feats );
                writeMatrix( cfgData.outFileName + "-trainNegFeats.bin", feats );
            }
        }

        /** Finish with sigmoid fit **/
        {
            struct timespec ts1, ts2;

            Eigen::ArrayXf trainedScoreFloat;
            clock_gettime( CLOCK_REALTIME, &ts1 );
            adaboost.predict( adaboost.getSampleIdxs(), trainedScoreFloat, trainCombo, true );
            clock_gettime( CLOCK_REALTIME, &ts2 );
            //float t_p = (float) ( 1.0*(1.0*ts2.tv_nsec - ts1.tv_nsec*1.0)*1e-9 + 1.0*ts2.tv_sec - 1.0*ts1.tv_sec );
            //int t_h = floor(t_p / 3600.);
            //t_p -= 3600. * t_h;
            //int t_m = floor(t_p / 60.);
            //t_p -= 60. * t_m;
            //int t_s = round(t_p);
            //printf("CCboost service :: Prediction: %d:%02d:%02d\n", t_h, t_m, t_s);

            Eigen::ArrayXd trainedScore = trainedScoreFloat.cast<double>();

            const std::vector<unsigned char> &minClassIdxs = adaboost.getMinorityClassIdxs();

            unsigned int numPos = 0;
            unsigned int numNeg = 0;
            for (unsigned i=0; i < (unsigned)trainedScore.rows(); i++)
            {
                if (minClassIdxs[i] != 0)
                    numPos++;
                else
                    numNeg++;
            }

            Eigen::ArrayXd sampleWeights;
            sampleWeights.resize( trainedScore.rows() );
            const float posWeight = numNeg * 1.0 / (numPos + numNeg);
            const float negWeight = numPos * 1.0 / (numPos + numNeg);
            for (unsigned i=0; i < trainedScore.rows(); i++)
            {
                if (minClassIdxs[i] != 0)
                    sampleWeights.coeffRef(i) = posWeight;
                else
                    sampleWeights.coeffRef(i) = negWeight;
            }

            //sampleWeights = sampleWeights / sampleWeights.sum();


            /*FILE *f = fopen("data.txt", "w");
            for (unsigned i=0; i < trainedScore.rows(); i++)
                fprintf(f, "%d %f %f\n", adaboost.getSampleClasses()[i], trainedScore[i], sampleWeights[i]);

            fclose(f);
            qDebug("File written");*/

            if (cfgData.outputPlattScaling)
            {
                qPrint("Learning platt scaling factors");
                sigmoidPlatt.Fit( trainedScore, adaboost.getSampleClasses(), &sampleWeights );
            }

            qPrint("Platt: %f  /  %f %d %d", sigmoidPlatt.getA(), sigmoidPlatt.getB(), (int) numNeg, (int) numPos);
        }

        // save learned stumps + platt scaling, only if it was not saved before
        if ( cfgData.savedStumpsPath.empty() )
            adaboost.saveLearnedStumps( cfgData.outFileName + "-stumps.cfg", &sigmoidPlatt );

        // save weight images?
        if (cfgData.saveTrainingWeights)
        {
            qPrint("Saving weight images");
            saveTrainingWeights( cfgData, adaboost, trainCombo, appendedPositivesAsNegativesIdx );
        }

    }

    /** ------- TEST PHASE ***********/
    qPrint("STARTING TESTING!");

    for (unsigned ti = 0; ti < cfgData.test.size(); ti++)
    {
        SVCombo testCombo;
        computeSupervoxelCombo( cfgData.test[ti], svStep, svCubeness, testCombo );

        const std::string outputBaseName = cfgData.outFileName + "-" + xToString(ti) + "-";

        // create some references to ease notation
        SuperVoxeler<PixelType> &SVox = testCombo.SVox;
        std::vector<UIntPoint3D> &svCentroids = testCombo.svCentroids;
        const Matrix3D<PixelType> &rawImage = testCombo.rawImage;

        std::vector<Eigen::Matrix3f> &rotMatrices = testCombo.rotMatrices;

        std::vector<unsigned int> predSamples( SVox.numLabels() );
        for (unsigned int i=0; i < predSamples.size(); i++)
            predSamples[i] = i;

        Eigen::ArrayXf  score, scoreInv;

        HistogramMeanThresholdData params( svCentroids, testCombo.rotMatrices, testCombo.pixIntegralImages, SVox.pixelToVoxel(),
                                           cfgData.test[ti].zAnisotropyFactor
#if APPLY_PATCH_VARNORM
                                           ,testCombo.svoxWindowInvStd, testCombo.svoxWindowMean);
#else
                                         );
#endif

        adaboost.reInit( params );  // set new params for test volume

        {


            // should we save features?
            bool outputFeatureData = false;
            if (cfgData.keyExists("extra.saveTestingFeatures"))
            {
                bool doIt;
                cfgData.getKeyValue("extra.saveTestingFeatures", doIt);

                if (doIt)
                    outputFeatureData = true;
            }

            {
                qPrint("Starting prediction");
                TimerRT predTimer;
                predTimer.Reset();
                adaboost.predict( predSamples, score, testCombo, cfgData.outputPlattScaling );
                //qDebug("Pred elapsed: %f", predTimer.elapsed());

                if (cfgData.outputPlattScaling)
                    sigmoidPlatt.Apply( score, score );


                Eigen::ArrayXXf eFeats;

                // save features?
                if ( outputFeatureData )
                {
                    qDebug("Generating features (REQUESTED IN CONFIG FILE)");

                    adaboost.exportFeatures( predSamples, eFeats );

                    qDebug("Saving features");
                    {
                        qDebug("Test matrix size: %d x %d", (int)eFeats.rows(), (int)eFeats.cols() );
                        writeMatrix( outputBaseName + "-testFeaturesA.bin", eFeats );
                    }
                }

                testCombo.invertRotMatrices();

                predTimer.Reset();
                adaboost.predict( predSamples, scoreInv, testCombo, cfgData.outputPlattScaling );
                //qDebug("Pred elapsed: %f", predTimer.elapsed());

                float t_p = predTimer.elapsed();
                int t_h = floor(t_p / 3600.);
                t_p -= 3600. * t_h;
                int t_m = floor(t_p / 60.);
                t_p -= 60. * t_m;
                int t_s = round(t_p);
                qPrint("Done: %d:%02d:%02d s.", t_h, t_m, t_s);

                if (cfgData.outputPlattScaling)
                    sigmoidPlatt.Apply( scoreInv, scoreInv );

                Eigen::ArrayXXf eFeatsInv;
                Eigen::ArrayXXf eFeatsMaxResp;

                // save features?
                if ( outputFeatureData )
                {
                    qDebug("Generating features (REQUESTED IN CONFIG FILE)");

                    adaboost.exportFeatures( predSamples, eFeatsInv );

                    qDebug("Saving features");
                    {
                        writeMatrix( outputBaseName + "-testFeaturesB.bin", eFeatsInv );
                    }

                    eFeatsMaxResp.resizeLike(eFeatsInv);
                }

                // create matrix and write to file
                {
                    typedef itk::VectorImage<float, 3>  ItkVectorImageType;
                    ItkVectorImageType::Pointer predOrientImg;  // will be filled only if requested

                    if ( cfgData.outputOrientationEstimate )
                    {
                        itk::ImageFileReader<ItkVectorImageType>::Pointer reader = itk::ImageFileReader<ItkVectorImageType>::New();
                        reader->SetFileName( cfgData.test[ti].orientEstimate );
                        reader->Update();

                        predOrientImg = reader->GetOutput();
                    }


                    Matrix3D<PixelType> predVolumeMax, predVolumeMin;
                    Matrix3D<float>     predVolumeMaxF, predVolumeMinF;

                    if (cfgData.outputPlattScaling)
                    {
                        predVolumeMax.reallocSizeLike( rawImage );
                        predVolumeMin.reallocSizeLike( rawImage );
                    }
                    else
                    {
                        predVolumeMaxF.reallocSizeLike( rawImage );
                        predVolumeMinF.reallocSizeLike( rawImage );
                    }


                    for (unsigned int i=0; i < predSamples.size(); i++)
                    {
                        float minS = score(i);
                        float maxS = scoreInv(i);

                        if (outputFeatureData)
                            eFeatsMaxResp.row(i) = eFeatsInv.row(i);

                        bool voteForInv = true;

                        if (minS > maxS)
                        {
                            float tmp = minS;
                            minS = maxS;
                            maxS = tmp;

                            if (outputFeatureData)
                                eFeatsMaxResp.row(i) = eFeats.row(i);

                            voteForInv = false;
                        }

                        double probMax = 0;
                        double probMin = 0;
                        if (false)
                        {
                            probMax = 1.0/(1.0 + exp(-2*maxS));
                            probMin = 1.0/(1.0 + exp(-2*minS));
                        }
                        else
                        {
                            probMax = maxS;
                            probMin = minS;
                        }

                        const PixelInfoList &pixList = SVox.voxelToPixel().at(predSamples[i]);

                        // then we have to invert the orientation of the voxels inside this supervoxel
                        if (cfgData.outputOrientationEstimate && voteForInv)
                        {
                            for (unsigned int p=0; p < pixList.size(); p++)
                            {
                                ItkVectorImageType::IndexType index;
                                index[0] = pixList[p].coords.x;
                                index[1] = pixList[p].coords.y;
                                index[2] = pixList[p].coords.z;

                                predOrientImg->SetPixel( index, - predOrientImg->GetPixel(index) );
                            }
                        }

                        if (cfgData.outputPlattScaling)
                        {
                            const PixelType valMax = 254 * probMax;
                            const PixelType valMin = 254 * probMin;

                            for (unsigned int p=0; p < pixList.size(); p++)
                            {
                                predVolumeMax.data()[ pixList[p].index ] = valMax;
                                predVolumeMin.data()[ pixList[p].index ] = valMin;
                            }
                        }
                        else
                        {
                            for (unsigned int p=0; p < pixList.size(); p++)
                            {
                                predVolumeMaxF.data()[ pixList[p].index ] = probMax;
                                predVolumeMinF.data()[ pixList[p].index ] = probMin;
                            }
                        }
                    }

                    if (cfgData.outputOrientationEstimate)
                    {
                        qPrint("Saving orientation estimate (should not be here)");
                        std::string fName = outputBaseName + "-predOrient.nrrd";

                        itk::ImageFileWriter<ItkVectorImageType>::Pointer writer = itk::ImageFileWriter<ItkVectorImageType>::New();
                        writer->SetFileName( fName );
                        writer->SetInput( predOrientImg );
                        writer->Update();
                    }

                    if (outputFeatureData)
                    {
                        // save
                        qPrint("Saving features (should not be here)");
                        {
                            writeMatrix( outputBaseName + "testFeatures-max.bin", eFeatsMaxResp );
                        }
                    }

                    if (cfgData.outputPlattScaling)
                    {
                        predVolumeMax.save( outputBaseName + "ab-max.tif" );
                        predVolumeMin.save( outputBaseName + "ab-min.tif" );
                    }
                    else
                    {
                        predVolumeMaxF.save( outputBaseName + "ab-max.nrrd" );
                        predVolumeMinF.save( outputBaseName + "ab-min.nrrd" );
                    }
                }
            }
        }
    }

    qPrint("CCBOOST Finished!");

    return 0;
}

