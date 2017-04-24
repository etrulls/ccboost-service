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

#include "ConfigData.h"

#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <fstream>

#define qFatal(...) do { fprintf(stderr, "ERROR: "); fprintf (stderr, __VA_ARGS__); fprintf(stderr, "\n");  exit(-1); } while(0)
#define qDebug(...) do { fprintf (stdout, __VA_ARGS__); fprintf(stdout, "\n"); fflush(stdout); } while(0)



static bool fexists(const std::string &filename)
{
    std::ifstream ifile(filename.c_str(), std::ifstream::in);
    ///ifile.open();
    return ifile.good();
}

static void parseSetConfig( const libconfig::Config &cfg, const std::string &groupName, SetConfigData &out, bool checkFilesExist, bool noGTneeded = false )
{
#define mymacro(x)  \
    do { \
        cfg.lookupValue( groupName + "." + #x, out.x ); \
        if (!fexists(out.x))    qFatal("File not found: %s / %s", out.x.c_str(), std::string(#x).c_str()); \
    } while(0)

    mymacro(rawVolume);

    if (!noGTneeded) {
        mymacro(groundTruth);
        out.groundTruthErodeRadius = cfg.lookup( groupName + ".groundTruthErodeRadius");
    } else {
        out.groundTruth = "N/A";
        out.groundTruthErodeRadius = 0;
    }

    out.zAnisotropyFactor = cfg.lookup( groupName + ".zAnisotropyFactor");
    
    if (out.zAnisotropyFactor < 1.0)
        qFatal("Anisotropy factor less than 1.0, almost surely you made a mistake.");
    
    mymacro(orientEstimate);

    const libconfig::Setting& s = cfg.lookup(groupName + "." + "otherFeatures");

    out.otherFeatures.clear();
    for (unsigned i=0; i < (unsigned)s.getLength(); i++ ) {
        out.otherFeatures.push_back( s[i] );
        if (!fexists(out.otherFeatures[i]))    qFatal("File not found: %s", out.otherFeatures[i].c_str());
    }
    
    

    if ( !noGTneeded )
    {
        // load CC GT data
        const libconfig::Setting& s = cfg.lookup(groupName + "." + "ccPolarity");

        out.ccGTOrient.clear();

        if (s.getLength() == 0)
            qFatal("Config Error: ccPolarity cannot be empty!");

        for (unsigned i=0; i < (unsigned)s.getLength(); i++)
        {
            char str[1024];
            unsigned pos[3]; float vec[3];

            for (unsigned q=0; q < 3; q++)
            {
                sprintf(str, "%s.ccPolarity.[%d].pos", groupName.c_str(), i, q);
                pos[q] = cfg.lookup( str )[q];

                sprintf(str, "%s.ccPolarity.[%d].polarity", groupName.c_str(), i, q);
                vec[q] = (float)(int)cfg.lookup( str )[q];

                if (vec[q] != 1.0f && vec[q] != -1.0f && vec[q] != 0.0f)
                    qFatal("Config Error: in ccPolarity, valid values are -1, 1 or 0. Found: %f", vec[q]);
            }

            ConfigGTCCOrient data;
            data.px = pos[0];
            data.py = pos[1];
            data.pz = pos[2];

            data.vx = vec[0];
            data.vy = vec[1];
            data.vz = vec[2];

            out.ccGTOrient.push_back(data);
        }
    }

#undef mymacro
}

bool ConfigData::keyExists( const std::string &keyName )
{
	return cfg.exists(keyName);
}


static bool appendFileContentsToString( const std::string &fName, std::string &dest )
{
    std::ifstream inFile( fName.c_str() );
    if (!inFile.is_open())
    {
        qFatal("Error reading file: %s", fName.c_str());
        return false;
    }

    const std::string fileContents  = std::string(std::istreambuf_iterator<char>( inFile ),
    std::istreambuf_iterator<char>( ));
    inFile.close();

    dest += fileContents;
}

void ConfigData::open( const std::string &fName, bool checkFilesExist, const std::string &extraOpts )
{
    test.clear();

    // read file first
    std::string allConfigStr = "";

    if ( !fName.empty() )
        appendFileContentsToString( fName, allConfigStr );

    if (!extraOpts.empty())
        allConfigStr += extraOpts;

    qDebug("---- CONFIG STRING ------");
    qDebug("%s", allConfigStr.c_str());

    cfg.readString( allConfigStr );


    // only read train info if savedStumpsPath is not found
    if ( cfg.lookupValue( "savedStumpsPath", savedStumpsPath ) )
    {
        usedSavedStumpsPathOnlyForTesting = cfg.lookup("usedSavedStumpsPathOnlyForTesting");
        if (!usedSavedStumpsPathOnlyForTesting)
            parseSetConfig(cfg, "train", train, checkFilesExist);
    }
    else {
        usedSavedStumpsPathOnlyForTesting = false;
        parseSetConfig(cfg, "train", train, checkFilesExist);
    }


    {
        const libconfig::Setting& s = cfg.lookup("test");
        for (int i=0; i < s.getLength(); i++) {
            char str[1024];
            sprintf(str, "test.[%d]", i);

            test.push_back( SetConfigData() );
            parseSetConfig(cfg, std::string(str), test[i], checkFilesExist, true);
        }
    }

    numStumps = cfg.lookup("numStumps");
	numWLToExplorePerIter = cfg.lookup("numWLToExplorePerIter");
	
	thresholdSearchOnWholeData = cfg.lookup("thresholdSearchOnWholeData");
	
    outFileName = (const char *)cfg.lookup("outFileName");

    svoxSeed = cfg.lookup("svox.seed");
    svoxCubeness = cfg.lookup("svox.cubeness");    

    appendArtificialNegatives = cfg.lookup("appendArtificialNegatives");

    if (appendArtificialNegatives)
        treatArtificialNegativesAsPositivesInResampling = cfg.lookup("treatArtificialNegativesAsPositivesInResampling");
    else
        treatArtificialNegativesAsPositivesInResampling = false;

    saveTrainingWeights = cfg.lookup("saveTrainingWeights");

    if (saveTrainingWeights) {
        saveTrainingWeightsNegFileName = (const char *) cfg.lookup("saveTrainingWeightsNegFileName");
        saveTrainingWeightsPosFileName = (const char *) cfg.lookup("saveTrainingWeightsPosFileName");
    }
    else {
        saveTrainingWeightsNegFileName = "";
        saveTrainingWeightsPosFileName = "";
    }

    {
        const libconfig::Setting& s = cfg.lookup("useSynapses");
        for (int i=0; i < s.getLength(); i++) {
            char str[1024];
            sprintf(str, "useSynapses.[%d]", i);

            unsigned val = cfg.lookup(str);
            useSynapses.push_back(val);
        }
    }

    borderMinDist = cfg.lookup("borderMinDist");
    outputPlattScaling = cfg.lookup("outputPlattScaling");

    reweightingSamplingRatio = cfg.lookup("reweightingSamplingRatio");
	
	minPosRegionSize = cfg.lookup("minPosRegionSize");

    relearnThresholds = cfg.lookup("relearnThresholds");

    if ( relearnThresholds && savedStumpsPath.empty() )
        qFatal("relearnThresholds = true but no saved stumps path provided!");

    outputOrientationEstimate = cfg.lookup("outputOrientationEstimate");
}
