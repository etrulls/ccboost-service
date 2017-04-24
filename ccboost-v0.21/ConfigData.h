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

#ifndef CONFIGDATA_H
#define CONFIGDATA_H

#include <string>
#include <vector>
#include <libconfig.hh>
#include <iostream>


// orientation for conn comp. in ground truth
struct ConfigGTCCOrient
{
    unsigned px, py, pz;    // position
    float    vx, vy, vz;    // only used for sign / zero
};

// for train/test
struct SetConfigData
{
    std::string rawVolume;

    std::string groundTruth;
    unsigned int groundTruthErodeRadius;    // if 0 => no erosion is performed

    std::string orientEstimate;

    std::vector< std::string > otherFeatures;


    // ground truth conn component orientation, only for GT
    std::vector<ConfigGTCCOrient>  ccGTOrient;
    
    // level of anisotropy of the z direction, for anisotropic stacks
    // 1.0 implies that it is isotropic, and k means that spacing in z is k times the spacing in x or y
    double zAnisotropyFactor;

    void printInfo()
    {
        std::cout << "Volume: " << rawVolume << std::endl;
        std::cout << "GT: " << groundTruth << std::endl;
        std::cout << "GT erode radius: " << groundTruthErodeRadius << std::endl;
        std::cout << "Orient: " << orientEstimate << std::endl;
        std::cout << "Z anisotropy factor: " << zAnisotropyFactor << std::endl;
        
        for (unsigned i=0; i < otherFeatures.size(); i++)
            std::cout << "Otherfeature[" << i << "]: " << otherFeatures[i] << std::endl;
    }
};

class ConfigData
{
public:
    SetConfigData   train;

    // in the case that training was already done..
    std::string savedStumpsPath;

    // synapses to use for training
    std::vector<unsigned int>   useSynapses;

    // use savedstumpspath only for testing
    //  otherwise it goes through training but
    //  doesn't train anything, it is only to export the training samples
    bool usedSavedStumpsPathOnlyForTesting;

    // there can be multiple test sets
    std::vector<SetConfigData> test;

	unsigned int numWLToExplorePerIter;
	
	bool thresholdSearchOnWholeData;

    unsigned int numStumps;
    std::string outFileName;

    unsigned int svoxSeed;
    unsigned int svoxCubeness;
	
	unsigned int minPosRegionSize;
	unsigned int borderMinDist;

    bool appendArtificialNegatives; // set to true to use proper pose indexing
    bool treatArtificialNegativesAsPositivesInResampling;
        // if true, 'artificial negatives' are ALWAYS included with the real positives
        // for each iteration while looking for a weaklearner

    // sampling ratio for reweighting by sampling
    unsigned int reweightingSamplingRatio;

    bool saveTrainingWeights;
    std::string saveTrainingWeightsPosFileName;
    std::string saveTrainingWeightsNegFileName;

    bool relearnThresholds;

    // true if platt scaling should be used as output,
    // otherwise it just outputs score in nrrd file
    bool outputPlattScaling;

    // if true, outputs an orientation volume where
    //  each voxel is assigned a vector, whose
    //  orientation is the same as the orientation volume given as input
    //  but the polarity is chosen according to the computed adaboost score
    bool outputOrientationEstimate;



private:
    libconfig::Config	cfg;
	
public:
    void open( const std::string &fName, bool checkFilesExist = true, const std::string &extraOpts = "");
    
    bool keyExists( const std::string &keyName );
    
    template<typename T>
    inline void getKeyValue( const std::string &keyName, T &value )
    {
            cfg.lookupValue( keyName, value );
    }

    template<typename T>
    inline T getKeyValue( const std::string &keyName )
    {
        return cfg.lookup( keyName );
    }

    void printInfo()
    {
        std::cout << "---- TRAIN -----" << std::endl;
        train.printInfo();

        for (unsigned i=0; i < test.size(); i++)
        {
            std::cout << std::endl << "---- TEST " << i << "-----" << std::endl;
            test[i].printInfo();
        }

        std::cout << "SVox seed: " << svoxSeed <<  std::endl;
        std::cout << "SVox cubeness: " << svoxCubeness <<  std::endl;
        std::cout << "Num stumps: " << numStumps << std::endl;
        std::cout << "Outfilename: " << outFileName << std::endl;
        std::cout << "Append artif. negatives: " << appendArtificialNegatives << std::endl;
        std::cout << "Treat artif. negatives as pos during resampling: " << treatArtificialNegativesAsPositivesInResampling << std::endl;
        std::cout << "Platt scaling output: " << outputPlattScaling << std::endl;
        std::cout << "Saved stumps path: " << savedStumpsPath << std::endl;
        std::cout << "usedSavedStumpsPathOnlyForTesting: " << usedSavedStumpsPathOnlyForTesting << std::endl;
        std::cout << "Reweighting sampling ratio: " << reweightingSamplingRatio << std::endl;
    }
};

#endif // CONFIGDATA_H
