//////////////////////////////////////////////////////////////////////////////////
//																																							//
// Copyright (C) 2011 Engin Turetken																						//
//																																							//
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
//                                                                              //
// Contact <engin.turetken@epfl.ch> for comments & bug reports                  //
//////////////////////////////////////////////////////////////////////////////////

#ifndef __SigmoidFitterPlatt_h
#define __SigmoidFitterPlatt_h

#include <vector>
#include <cmath>
#include <Eigen/Dense>

/** \class SigmoidFitterPlatt
 *
 * \brief Implements Levenberg-Marquardt based sigmoid fitting optimization as
             * described in the following paper. This can be used for estimating
             * probabilities from classifier scores in a two-class classification setting.
             *
             * Platt J., Advances in Large Margin Classifiers, MIT Press, Chap.
             * Probabilistic Outputs for SVMs and Comparisons to Regularized Likelihood
             * Methods, 2000.
 *
 * \author Engin Turetken
 * 
 * Modified by Carlos Becker, implementing weighted fitting from 
 * http://www.opensourcejavaphp.net/java/rapidminer/com/rapidminer/operator/postprocessing/PlattScaling.java.html
 * (RapidMiner source code in Java)
 *
 */
template<typename T>
class SigmoidFitterPlatt
{
private:
    T mA, mB;
    typedef T ScalarType;

public:
    inline T getA() const { return mA; }
    inline T getB() const { return mB; }
    
    SigmoidFitterPlatt()
	{
		// initialize to default values, such that it would be 1/(1+exp(-2*score))
		mA = -2;
		mB = 0;
	}

    void save( libconfig::Setting &s ) const
    {
        s.add("A", libconfig::Setting::TypeFloat) = mA;
        s.add("B", libconfig::Setting::TypeFloat) = mB;
    }

    void load( const libconfig::Setting &s )
    {
        mA = (double) s["A"];
        mB = (double) s["B"];
    }

    template<typename T2, typename T3>
    inline void Apply( const Eigen::Array<T2, Eigen::Dynamic, 1> &scores, Eigen::Array<T3, Eigen::Dynamic, 1> &dest  )
    {
        //dest = ((ScalarType) 1) / ( ((ScalarType) 1) + (mA * scores + mB).exp() );
        dest = (( ((ScalarType) 1) + (mA * scores.template cast<double>() + mB).exp() ).inverse()).template cast<T3>();
    }

    /**
     * Given a set of examples with classifier responses and labels
     * (positive = 1 / negative = anything other than 1), this function estimates
     * the optimal sigmoid parameters that maps the classifier responses to
     * probablities. The sigmoid function is defined as
     *
     * p(x) = 1 / (1 + exp(Ax+B))
     */
    template<typename T2, typename T3>
    void Fit(const Eigen::Array<T, Eigen::Dynamic, 1>& responseArray,
                                     const std::vector<T2>& classLabelArray,
                                     const Eigen::Array<T3, Eigen::Dynamic, 1> *weightArray = 0 )
    {
        //Declerations
        int it;
        int* target;
        ScalarType prior0;
        ScalarType prior1;
        ScalarType hiTarget;
        ScalarType loTarget;
        ScalarType t;
        ScalarType lambda;
        ScalarType olderr;
        ScalarType err;
        ScalarType* pp;
        ScalarType ppInitVal;
        int count;
        ScalarType a, b, c, d, e, d1, d2;
        ScalarType oldA, oldB;
        ScalarType det;
        ScalarType p;
        ScalarType diff;
        ScalarType scale;
        unsigned int nNoOfExamples = responseArray.rows();
        
        const bool hasWeights = weightArray != 0;
        
        if (classLabelArray.size() != (unsigned int) responseArray.rows())
			qFatal("Class label array and response array must be of same size");
		if (hasWeights) {
			if (responseArray.rows() != weightArray->rows())
				qFatal("Weight array size and response array must be of same size");
		}

        prior0 = 0; // number of negative examples
        prior1 = 0; // number of positive examples
        target = new int[nNoOfExamples];

        pp = new ScalarType[nNoOfExamples];	// temp array to store current estimate of probability of examples
        // set all pp array elements to (prior1+1)/(prior0+prior1+2)
        for( unsigned int i = 0; i < nNoOfExamples; i++ )
        {
                ScalarType weight = hasWeights? weightArray->coeff(i) : 1;

                if( classLabelArray[i] != 1 )
                {
                        target[i] = 0;
                        prior0 += weight;
                }
                else
                {
                        target[i] = 1;
                        prior1 += weight;
                }
        }
        
        ppInitVal = (prior1+1)/(prior0+prior1+2);
        for( unsigned int i = 0; i < nNoOfExamples; i++ )
			pp[i] = ppInitVal;

        mA = 0;
        mB = log((prior0+1)/(prior1+1));
        hiTarget = (prior1+1)/(prior1+2);
        loTarget = 1/(prior0+2);
        lambda = 1e-3;
        olderr = 1e300;
        count = 0;
        for( it = 0; it < 1000; it++ )
        {
                a = 0, b = 0, c = 0, d = 0, e = 0;
                // First, compute Hessian & gradient of error function
                // with respect to A & B
                for( unsigned int i = 0; i < nNoOfExamples; i++ )
                {
						ScalarType weight = hasWeights? weightArray->coeff(i) : 1;
						
                        if( target[i] )
                        {
                                t = hiTarget;
                        }
                        else
                        {
                                t = loTarget;
                        }
                        d1 = (pp[i] - t) * weight;
                        d2 = (pp[i] * (1-pp[i])) * weight;
                        a += responseArray[i] * responseArray[i] * d2;
                        b += d2;
                        c += responseArray[i] * d2;
                        d += responseArray[i] * d1;
                        e += d1;
                }

                // If gradient is really tiny, then stop
                if( fabs(d) < 1e-9 && fabs(e) < 1e-9 )
                {
                        break;
                }

                oldA = mA;
                oldB = mB;
                err = 0;
                // Loop until goodness of fit increases
                while (1)
                {
                        det = (a+lambda)*(b+lambda)-c*c;
                        if( det == 0 ) // if determinant of Hessian is zero, increase stabilizer.
                        {
                                lambda *= 10;
                                continue;
                        }
                        mA = oldA + ((b+lambda)*d-c*e)/det;
                        mB = oldB + ((a+lambda)*e-c*d)/det;
                        // Now, compute the goodness of fit
                        err = 0;
                        for( unsigned int i = 0; i < nNoOfExamples; i++ )
                        {
                                p = 1/(1+exp(responseArray[i]*mA+mB));
                                pp[i] = p;
                                
                                ScalarType weight = hasWeights? weightArray->coeff(i) : 1;
                                // At this step, make sure log(0) returns -200
                                err -= weight * ( t*log(p)+(1-t)*log(1-p) );
                        }
                        if(err < olderr*(1+1e-7))
                        {
                                lambda *= 0.1;
                                break;
                        }
                        // error did not decrease: increase stabilizer by factor of 10
                        // & try again
                        lambda *= 10;
                        if( lambda >= 1e6 ) // something is broken. Give up
                        {
                                break;
                        }
                }
                diff = err-olderr;
                scale = 0.5*(err+olderr+1);
                if( (diff > (-1e-3 * scale)) && (diff < (1e-7*scale)) )
                {
                        count++;
                }
                else
                {
                        count = 0;
                }
                olderr = err;
                if( count == 3 )
                {
                        break;
                }
        }

        //Deallocations
        delete[] target;
        delete[] pp;

    }
};

#endif
