///////////////////////////////////////////////////////////
//  HSC_SolverComps.cpp
//  Implementation of the Class HSC_SolverComps
//  Created on:      27-Dec-2013 7:57:50 PM
//  Original author: Saeed Shariati
///////////////////////////////////////////////////////////

#include "HSC_SolverComps.h"
#include <assert.h>
#include <cublas_v2.h>
#include "solve.h"

// Macro to catch CUDA errors in CUDA runtime calls
#define CUDA_SAFE_CALL(call)                                          \
do {                                                                  \
    cudaError_t err = call;                                           \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString(err) );       \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
} while (0)

cublasHandle_t cublasHandle;

template <typename T, int arch>
HSC_SolverComps<T,arch>::HSC_SolverComps()
{
	_hmListOfHost = 0;
	_hmListOfMatricesInDevice = 0;
	_pivotArray_h = 0;
	_hmList_dev = 0;
	_infoArray_h = 0;
	_infoArray_d = 0;
	_pivotArray_d = 0;
}

template <typename T, int arch>
HSC_SolverComps<T,arch>::~HSC_SolverComps()
{
	if (_pivotArray_h) free(_pivotArray_h);
	if(_hmListOfHost) free(_hmListOfHost[0]);
	if(_hmListOfMatricesInDevice) cudaFree(_hmListOfMatricesInDevice[0]);
	if (_hmListOfMatricesInDevice) free(_hmListOfMatricesInDevice);
	if (_hmList_dev) cudaFree(_hmList_dev);
	if (_infoArray_h) free(_infoArray_h);
	if (_infoArray_d) cudaFree(_infoArray_d);
	if (_pivotArray_d) cudaFree(_pivotArray_d);
}

template <typename T, int arch>
hscError HSC_SolverComps<T,arch>::PrepareSolver(vector<HSCModel > &network, HSC_NetworkAnalyzer &analyzer)
{
	nModel = analyzer.nModel;
	nComp = analyzer.nComp;

	cudaError_t success;

	_hmListOfMatricesInDevice =(T **)malloc(nModel * sizeof(*_hmListOfMatricesInDevice));
	_hmListOfHost =(T **)malloc(nModel * sizeof(*_hmListOfHost));

	//Get memory for each hm in GPU and Host
	success = cudaMalloc((void **)&_hmListOfMatricesInDevice[0],nModel* nComp*nComp * sizeof(_hmListOfMatricesInDevice[0][0]));
	_hmListOfHost[0] = (T *)malloc(nModel* nComp*nComp * sizeof(_hmListOfHost[0][0]));
	assert(!success);
	for (int i = 1; i < nModel; ++i) {
		_hmListOfHost[i] = _hmListOfHost[i-1]+nComp*nComp;
		_hmListOfMatricesInDevice[i] = _hmListOfMatricesInDevice[i-1]+nComp*nComp;
	}

	// Send pointer list to device
	_hmList_dev = NULL;
	success = cudaMalloc((void **)&_hmList_dev, nModel * sizeof(*_hmListOfMatricesInDevice));
	assert(!success);
	success = cudaMemcpy(_hmList_dev, _hmListOfMatricesInDevice, nModel * sizeof(*_hmListOfMatricesInDevice), cudaMemcpyHostToDevice);
	assert(!success);

	//A nComp x nModel matrix which contains PivotArray of all models in the network
	_pivotArray_h =(int *)malloc(nComp * nModel * sizeof(*_pivotArray_h));
	_infoArray_h =(int *)malloc(nModel * sizeof(*_infoArray_h));

	cudaMalloc((void**) (&_pivotArray_d), nModel*nComp * sizeof(*_pivotArray_d));
	cudaMalloc((void**) (&_infoArray_d), nModel * sizeof(*_infoArray_d));

	cublasStatus_t stat = cublasCreate(&cublasHandle);

	//making Hines Matrices
	//TODO: use streams
	for(int i=0; i< nModel;i++ )
	{
		makeHinesMatrix(&network[i], _hmListOfHost[i]);
		_printMatrix_Column(nComp,nComp, _hmListOfHost[i]);
		cublasSetMatrix(nComp, nComp, sizeof(_hmListOfHost[i][0]), _hmListOfHost[i], nComp, _hmListOfMatricesInDevice[i], nComp);
	}

	cudaDeviceSynchronize();

	cublasStatus_t cubSucc = cublasSgetrfBatched(
			cublasHandle, nComp,
			(float **)_hmList_dev, nComp,
			_pivotArray_d,
			_infoArray_d,
			nModel);
	assert(!cubSucc);

	cublasGetMatrix(nComp, nModel, sizeof(*_pivotArray_d), _pivotArray_d, nComp,_pivotArray_h, nComp);
	cublasGetVector(nModel, sizeof(*_infoArray_h), _infoArray_d, 1,_infoArray_h, 1);


	_printMatrix(nComp,nModel, _pivotArray_h);
	_printMatrix(nModel,1, _infoArray_h);

	memset(_hmListOfHost[0],0,nModel* nComp*nComp * sizeof(_hmListOfHost[0][0]));
	for(int i=0; i< nModel;i++ )
	{
		cubSucc = cublasGetMatrix(nComp, nComp, sizeof(_hmListOfHost[i][0]), _hmListOfMatricesInDevice[i], nComp,_hmListOfHost[i], nComp);
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

	return NO_ERROR;
}


template <typename T, int arch>
void HSC_SolverComps<T,arch>::makeHinesMatrix(HSCModel *model, T * matrix)
{
	unsigned int nCompt = model->compts.size();

	/*
	 * Some convenience variables
	 */
	vector< double > CmByDt(nCompt);
	vector< double > Ga(nCompt);
	for ( unsigned int i = 0; i < nCompt; i++ ) {
		CmByDt[i] = model->compts[ i ].Cm / ( _dt / 2.0 ) ;
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


template <typename T, int arch>
hscError HSC_SolverComps<T,arch>::Process()
{
	uint N = 3;
	uint BATCH = 20;
    double A[BATCH*N*N];
    double b[BATCH*N];
    double x[BATCH*N];
    double *A_d;
    double *b_d;
    double *x_d;

    CUDA_SAFE_CALL (cudaMalloc ((void**)&A_d, BATCH*N*N*sizeof(A_d[0])));
    CUDA_SAFE_CALL (cudaMalloc ((void**)&b_d, BATCH*N*sizeof(b_d[0])));
    CUDA_SAFE_CALL (cudaMalloc ((void**)&x_d, BATCH*N*sizeof(x_d[0])));

    CUDA_SAFE_CALL (cudaMemcpy (A_d, A, BATCH*N*N*sizeof(A_d[0]),
                                cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL (cudaMemcpy (b_d, b, BATCH*N*sizeof(b_d[0]),
                                cudaMemcpyHostToDevice));
    dsolve_batch (A_d, b_d, x_d, N, BATCH);
    CUDA_SAFE_CALL (cudaMemcpy (x, x_d, BATCH*N*sizeof(x_d[0]),
                                cudaMemcpyDeviceToHost));


    CUDA_SAFE_CALL(cudaFree(A_d));
    CUDA_SAFE_CALL(cudaFree(b_d));
    CUDA_SAFE_CALL(cudaFree(x_d));
	return NO_ERROR;
}

template class HSC_SolverComps<double, ARCH_SM30>;
