///////////////////////////////////////////////////////////
//  HSC_SolverComps.cpp
//  Implementation of the Class HSC_SolverComps
//  Created on:      27-Dec-2013 7:57:50 PM
//  Original author: Saeed Shariati
///////////////////////////////////////////////////////////

#include "HSC_SolverComps.h"
#include <assert.h>
#include <cublas_v2.h>

HSC_SolverComps::HSC_SolverComps(double _dt): dt(_dt)
{
	_hmListOfHost = 0;
	_hmListOfDevice = 0;
	_pivotArray_h = 0;
}

hscError HSC_SolverComps::PrepareSolver(vector<HSCModel > &network, HSC_NetworkAnalyzer &analyzer)
{
	uint nModel = analyzer.nModel;
	uint nComp = analyzer.nComp;

	cudaError_t success;

	_hmListOfDevice =(float **)malloc(nModel * sizeof(*_hmListOfDevice));
	_hmListOfHost =(float **)malloc(nModel * sizeof(*_hmListOfHost));

	//Get memory for each hm in GPU and Host
	success = cudaMalloc((void **)&_hmListOfDevice[0],nModel* nComp*nComp * sizeof(_hmListOfDevice[0][0]));
	_hmListOfHost[0] = (float *)malloc(nModel* nComp*nComp * sizeof(_hmListOfHost[0][0]));
	assert(!success);
	for (int i = 1; i < nModel; ++i) {
		_hmListOfHost[i] = _hmListOfHost[i-1]+nComp*nComp;
		_hmListOfDevice[i] = _hmListOfDevice[i-1]+nComp*nComp;
	}

	// Send pointer list to device
	float **_hmList_dev = NULL;
	success = cudaMalloc((void **)&_hmList_dev, nModel * sizeof(*_hmListOfDevice));
	assert(!success);
	success = cudaMemcpy(_hmList_dev, _hmListOfDevice, nModel * sizeof(*_hmListOfDevice), cudaMemcpyHostToDevice);
	assert(!success);

	//Create output variables
	int * infoArray_h = 0;

	//A nComp x nModel matrix which contains PivotArray of all models in the network
	_pivotArray_h =(int *)malloc(nComp * nModel * sizeof(*_pivotArray_h));
	infoArray_h =(int *)malloc(nModel * sizeof(*infoArray_h));

	int * pivotArray_d;
	int * infoArray_d;
	cudaMalloc((void**) (&pivotArray_d), nModel*nComp * sizeof(*pivotArray_d));
	cudaMalloc((void**) (&infoArray_d), nModel * sizeof(*infoArray_d));

	cublasHandle_t cublasHandle;
	cublasStatus_t stat = cublasCreate(&cublasHandle);

	//making Hines Matrices
	//TODO: use streams
	for(int i=0; i< nModel;i++ )
	{
		makeHinesMatrix(&network[i], _hmListOfHost[i]);
		_printMatrix_Column(nComp,nComp, _hmListOfHost[i]);
		cublasSetMatrix(nComp, nComp, sizeof(_hmListOfHost[i][0]), _hmListOfHost[i], nComp, _hmListOfDevice[i], nComp);
	}

	cudaDeviceSynchronize();

	cublasStatus_t cubSucc = cublasSgetrfBatched(
			cublasHandle, nComp,
			(float **)_hmList_dev, nComp,
			pivotArray_d,
			infoArray_d,
			nModel);
	assert(!cubSucc);

	cublasGetMatrix(nComp, nModel, sizeof(*pivotArray_d), pivotArray_d, nComp,_pivotArray_h, nComp);
	cublasGetVector(nModel, sizeof(*infoArray_h), infoArray_d, 1,infoArray_h, 1);


	_printMatrix(nComp,nModel, _pivotArray_h);
	_printMatrix(nModel,1, infoArray_h);

	memset(_hmListOfHost[0],0,nModel* nComp*nComp * sizeof(_hmListOfHost[0][0]));
	for(int i=0; i< nModel;i++ )
	{
		cubSucc = cublasGetMatrix(nComp, nComp, sizeof(_hmListOfHost[i][0]), _hmListOfDevice[i], nComp,_hmListOfHost[i], nComp);
		assert(!cubSucc);
		_printMatrix_Column(nComp,nComp, _hmListOfHost[i]);
	}

//
//	printMatrix_Column(n, n, A_h);
//	printMatrix(n, nModel, pivotArray_h);
//	printMatrix(nModel, 1, infoArray_h);
//
//	cublasDestroy(cublasHandle);
//

//	free(A_h);

	if (_pivotArray_h) free(_pivotArray_h);
	if(_hmListOfDevice[0]) cudaFree(_hmListOfDevice[0]);
	if(_hmListOfHost[0]) free(_hmListOfHost[0]);

	if (_hmListOfDevice) free(_hmListOfDevice);
	if (infoArray_h) free(infoArray_h);
	if (_hmList_dev) cudaFree(_hmList_dev);
	if (infoArray_d) cudaFree(infoArray_d);
	if (pivotArray_d) cudaFree(pivotArray_d);

	return NO_ERROR;
}


void HSC_SolverComps::makeHinesMatrix(HSCModel *model, float * matrix)
{
	unsigned int nCompt = model->compts.size();

	/*
	 * Some convenience variables
	 */
	vector< double > CmByDt(nCompt);
	vector< double > Ga(nCompt);
	for ( unsigned int i = 0; i < nCompt; i++ ) {
		CmByDt[i] = model->compts[ i ].Cm / ( dt / 2.0 ) ;
		Ga[i] =  2.0 / model->compts[ i ].Ra ;
	}

	/* Each entry in 'coupled' is a list of electrically coupled compartments.
	 * These compartments could be linked at junctions, or even in linear segments
	 * of the cell.
	 */
	vector< vector< unsigned int > > coupled;
	for ( unsigned int i = 0; i < nCompt; i++ )
		if ( model->compts[ i ].children.size() >= 1 ) {
			coupled.push_back( model->compts[ i ].children );
			coupled.back().push_back( i );
		}

	// Setting diagonal elements
	for ( unsigned int i = 0; i < nCompt; i++ )
		matrix[ i * nCompt + i ] = CmByDt[ i ] + 1.0 / model->compts[ i ].Rm;


	double gi;
	vector< vector< unsigned int > >::iterator group;
	vector< unsigned int >::iterator ic;
	for ( group = coupled.begin(); group != coupled.end(); ++group ) {
		double gsum = 0.0;

		for ( ic = group->begin(); ic != group->end(); ++ic )
			gsum += Ga[ *ic ];

		for ( ic = group->begin(); ic != group->end(); ++ic ) {
			gi = Ga[ *ic ];

			matrix[ *ic * nCompt + *ic ] += gi * ( 1.0 - gi / gsum );
		}
	}


	// Setting off-diagonal elements
	double gij;
	vector< unsigned int >::iterator jc;
	for ( group = coupled.begin(); group != coupled.end(); ++group ) {
		double gsum = 0.0;

		for ( ic = group->begin(); ic != group->end(); ++ic )
			gsum += Ga[ *ic ];

		for ( ic = group->begin(); ic != group->end() - 1; ++ic ) {
			for ( jc = ic + 1; jc != group->end(); ++jc ) {
				gij = Ga[ *ic ] * Ga[ *jc ] / gsum;

				matrix[ *ic * nCompt + *jc ] = -gij;
				matrix[ *jc * nCompt + *ic ] = -gij;
			}
		}
	}
}
