#!/usr/bin/python

import os

# imgFName = "volume"
imgFName = "/cvlabdata1/cvlab/datasets_eduard/em/source/carlos/hippocampus/train/images/volume.tif"
imgExt = '.tif'
nrrdExt = '.nrrd'

gaussBin = './GaussianImageFilter'
gradBin = './GradientMagnitudeImageFilter'
LoGBin = './LoGImageFilter'
eigOfHessBin = './EigenOfHessianImageFilter' 
eigOfST = './EigenOfStructureTensorImageFilter'
singleEigVecHess = './SingleEigenVectorOfHessian'
allEigVecHess = './AllEigenVectorsOfHessian' 
repolarizeOrient = './RepolarizeYVersorWithGradient'

outDir = 'synapse_features'
os.mkdir(outDir)

# this means isotropic if = 1.0
zAnisotropyFactor = "1.0"

if False:
	for sigma in [1.0, 1.6, 3.5, 5.0]:
		s = eigOfHessBin + ' ' + imgFName + imgExt + ' ' + str(sigma) + ' ' + zAnisotropyFactor + ' ' + outDir + '/hessian-s' + ('%.1f' % sigma) + nrrdExt + " 1"
		print s
		os.system( s )

if True:
	s = eigOfST + ' ' + imgFName + imgExt + ' 0.5 1.0 ' + zAnisotropyFactor + ' ' + outDir + '/stensor-s0.5-r1.0.nrrd' + ' 1'
	print s
	os.system( s )

	s = eigOfST + ' ' + imgFName + imgExt + ' 0.8 1.6 ' + zAnisotropyFactor + ' ' + outDir + '/stensor-s0.8-r1.6.nrrd' + ' 1'
	print s
	os.system( s )

	s = eigOfST + ' ' + imgFName + imgExt + ' 1.8 3.5 ' + zAnisotropyFactor + ' ' + outDir + '/stensor-s1.8-r3.5.nrrd' + ' 1'
	print s
	os.system( s )

	s = eigOfST + ' ' + imgFName + imgExt + ' 2.5 5.0 ' + zAnisotropyFactor + ' ' + outDir + '/stensor-s2.5-r5.0.nrrd' + ' 1'
	print s
	os.system( s )




print '---------------- XXXXXXXXXxx---'

if True:
	orientImgFName = outDir + '/hessOrient-s3.5-highestMag.nrrd'
	s = singleEigVecHess + ' ' + imgFName + imgExt + ' 3.5 ' + zAnisotropyFactor + ' ' + orientImgFName + ' 0'
	print s
	os.system(s)
	
if True:
	allEigVecHessFName = outDir + '/hessOrient-s3.5-allEigVecs.nrrd'
	s = allEigVecHess + ' ' + imgFName + imgExt + ' 3.5 ' + zAnisotropyFactor + ' ' + allEigVecHessFName + ' 1'
	print s
	os.system(s)

	s = repolarizeOrient + ' ' + imgFName + imgExt + ' ' + allEigVecHessFName + ' 3.5 ' + zAnisotropyFactor + ' ' + outDir + '/hessOrient-s3.5-repolarized.nrrd'
	print s
	os.system(s)

if True:
#	for sigma in [0.7, 1.0, 1.6, 3.5, 5.0]:
#		s = gaussBin + ' ' + imgFName + imgExt + ' ' + str(sigma) + ' ' + zAnisotropyFactor + ' ' + outDir + '/gauss-s' + ('%.1f' % sigma) + nrrdExt
#		print s
#		os.system( s )

	for sigma in [1.0, 1.6, 3.5, 5.0]:
		s = gradBin + ' ' + imgFName + imgExt + ' ' + str(sigma) + ' ' + zAnisotropyFactor + ' ' + outDir + '/gradient-magnitude-s' + ('%.1f' % sigma) + nrrdExt
		print s
		os.system( s )

#	for sigma in [1.0, 1.6, 3.5, 5.0]:
#		s = LoGBin + ' ' + imgFName + imgExt + ' ' + str(sigma) + ' ' + zAnisotropyFactor + ' ' + outDir + '/log-s' + ('%.1f' % sigma) + nrrdExt
#		print s
#		os.system( s )



print 'DONE'
