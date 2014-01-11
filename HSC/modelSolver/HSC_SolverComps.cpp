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
	_hm = 0;
	_hm_dev = 0;
	_rhs = 0;
	_rhs_dev = 0;
	_Vm = 0;
	_Vm_dev = 0;
}

template <typename T, int arch>
HSC_SolverComps<T,arch>::~HSC_SolverComps()
{
	if (_hm) free(_hm);
	if(_rhs) free(_rhs);
	if (_Vm) free(_Vm);
	if (_hm_dev) cudaFree(_hm_dev);
	if(_rhs_dev) cudaFree(_rhs_dev);
	if (_Vm_dev) cudaFree(_Vm_dev);
}

template <typename T, int arch>
hscError HSC_SolverComps<T,arch>::PrepareSolver(vector<HSCModel > &network, HSC_NetworkAnalyzer &analyzer)
{
	nModel = analyzer.nModel;
	nComp = analyzer.nComp;
	uint modelSize = nComp*nComp;
	cudaError_t success;
	cublasStatus_t cublasSuccess;
	//Define memory
	_hm =(T*)malloc(modelSize*nModel * sizeof(_hm[0]));
	CUDA_SAFE_CALL(cudaMallocHost((void **) &_rhs, nComp*nModel * sizeof(_rhs[0]))); //Define pinned memories
	CUDA_SAFE_CALL(cudaMallocHost((void **) &_Vm, nComp*nModel * sizeof(_Vm[0]))); 	//Define pinned memories

	CUDA_SAFE_CALL(cudaMalloc((void **) &_hm_dev, modelSize*nModel * sizeof(_hm_dev[0])));
	CUDA_SAFE_CALL(cudaMalloc((void **) &_rhs_dev, nComp*nModel * sizeof(_rhs_dev[0])));
	CUDA_SAFE_CALL(cudaMalloc((void **) &_Vm_dev, nComp*nModel * sizeof(_Vm_dev[0])));

	//making Hines Matrices
	for(int i=0; i< nModel;i++ )
	{
		makeHinesMatrix(&network[i], &_hm[i*modelSize]);
		_printMatrix_Column(nComp,nComp, &_hm[i*modelSize]);
		cublasSetMatrix(nComp, nComp, sizeof(_hm[0]),
				&_hm[i*modelSize], nComp,
				&_hm_dev[i*modelSize], nComp);
	}


	cudaDeviceSynchronize();

	//	cublasGetVector(nModel, sizeof(*_infoArray_h), _infoArray_d, 1,_infoArray_h, 1);

//	//Reset host mem
//	for(uint i=0; i< nModel*modelSize;i++ ) _hm[i]=0;
//	for(int i=0; i< nModel;i++ )
//	{
//		cublasSuccess = cublasGetMatrix(nComp, nComp, sizeof(_hm[0]),
//				&_hm_dev[i*modelSize], nComp,
//				&_hm[i*modelSize], nComp);
//		assert(!cublasSuccess);
//		_printMatrix_Column(nComp,nComp, &_hm[i*modelSize]);
//	}

	return NO_ERROR;
}

template <typename T, int arch>
hscError HSC_SolverComps<T,arch>::Process()
{
	for (int var = 0; var < nModel*nComp; ++var) {
		_rhs[var] = var*1000;
//		B[ i ] =V[ i ] * tree[ i ].Cm / ( dt / 2.0 ) +Em[ i ] / tree[ i ].Rm;
	}
	_printVector(nModel*nComp, _rhs);
	cublasSetVector(nModel*nComp, sizeof(_rhs[0]),_rhs, 1,_rhs_dev, 1);

    int ret = dsolve_batch (_hm_dev, _rhs_dev, _Vm_dev, nComp, nModel);
    assert(!ret);

//    for (int var = 0; var < nModel*nComp; ++var) {
//		_Vm[var] = 0;
//	}
	cublasGetVector(nModel*nComp, sizeof(_Vm[0]),_Vm_dev, 1,_Vm, 1);
//	_printVector(nModel*nComp, _Vm);

	return NO_ERROR;
}

template <typename T, int arch>
void HSC_SolverComps<T,arch>::makeHinesMatrix(HSCModel *model, T * matrix)
{
	/*
	 * Some convenience variables
	 */
	vector< double > CmByDt(nComp);
	vector< double > Ga(nComp);
	for ( unsigned int i = 0; i < nComp; i++ ) {
		CmByDt[i] = model->compts[ i ].Cm / ( _dt / 2.0 ) ;
		Ga[i] =  2.0 / model->compts[ i ].Ra ;
	}

	/* Each entry in 'coupled' is a list of electrically coupled compartments.
	 * These compartments could be linked at junctions, or even in linear segments
	 * of the cell.
	 */
	vector< vector< unsigned int > > coupled;
	for ( unsigned int i = 0; i < nComp; i++ )
		if ( model->compts[ i ].children.size() >= 1 ) {
			coupled.push_back( model->compts[ i ].children );
			coupled.back().push_back( i );
		}

	// Setting diagonal elements
	for ( unsigned int i = 0; i < nComp; i++ )
		matrix[ i * nComp + i ] = (T)(CmByDt[ i ] + 1.0 / model->compts[ i ].Rm);


	double gi;
	vector< vector< unsigned int > >::iterator group;
	vector< unsigned int >::iterator ic;
	for ( group = coupled.begin(); group != coupled.end(); ++group ) {
		double gsum = 0.0;

		for ( ic = group->begin(); ic != group->end(); ++ic )
			gsum += Ga[ *ic ];

		for ( ic = group->begin(); ic != group->end(); ++ic ) {
			gi = Ga[ *ic ];

			matrix[ *ic * nComp + *ic ] += (T) (gi * ( 1.0 - gi / gsum ));
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

				matrix[ *ic * nComp + *jc ] = (T)(-gij);
				matrix[ *jc * nComp + *ic ] = (T)(-gij);
			}
		}
	}
}

template class HSC_SolverComps<double, ARCH_SM30>;
