///////////////////////////////////////////////////////////
//  HSC_SolverComps.cpp
//  Implementation of the Class HSC_SolverComps
//  Created on:      27-Dec-2013 7:57:50 PM
//  Original author: Saeed Shariati
///////////////////////////////////////////////////////////

#include "HSC_SolverComps.h"
#include <assert.h>
#include <cuda_runtime.h>
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

template <typename T>
__inline__ hscError sendVector(uint size, T* h, T* d)
{
	cublasStatus_t stat = cublasSetVector(size, sizeof(h[0]),h, 1,d,1);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		cerr << "CUBLAS setting Variables\n";
		return CuBLASError;
	}
}

//CuBLAS variables
cublasHandle_t _handle;

//Thrust Variables
//thrust::device_vector<int> d_rhs = h_vec;


template <typename T, int arch>
HSC_SolverComps<T,arch>::HSC_SolverComps()
{
	_hm = 0;
	_hm_dev = 0;
	_rhs = 0;
	_rhs_dev = 0;
	_Vm = 0;
	_Vm_dev = 0;
	_Cm = 0;
	_Cm_dev = 0;
	_Em = 0;
	_Em_dev = 0;
	_Rm = 0;
	_Rm_dev = 0;
}

template <typename T, int arch>
HSC_SolverComps<T,arch>::~HSC_SolverComps()
{
	if (_hm) free(_hm);
	if(_rhs) free(_rhs);
	if (_Vm) free(_Vm);
	if (_Cm) free(_Cm);
	if (_Em) free(_Em);
	if (_Rm) free(_Rm);
	if (_hm_dev) cudaFree(_hm_dev);
	if(_rhs_dev) cudaFree(_rhs_dev);
	if (_Vm_dev) cudaFree(_Vm_dev);
	if (_Cm_dev) cudaFree(_Cm_dev);
	if (_Em_dev) cudaFree(_Em_dev);
	if (_Rm_dev) cudaFree(_Rm_dev);
}



template <typename T, int arch>
hscError HSC_SolverComps<T,arch>::PrepareSolver(vector<HSCModel > &network, HSC_NetworkAnalyzer &analyzer)
{
	cudaError_t success;
	cublasStatus_t stat;

	nModel = analyzer.nModel;
	nComp = analyzer.nComp;
	uint modelSize = nComp*nComp;
	uint vectorSize = nModel * nComp;

	//Define memory for host
	CUDA_SAFE_CALL(cudaMallocHost((void **) &_hm, modelSize*nModel * sizeof(_hm[0]))); //Define pinned memories
	CUDA_SAFE_CALL(cudaMallocHost((void **) &_rhs, vectorSize * sizeof(_rhs[0]))); //Define pinned memories
	CUDA_SAFE_CALL(cudaMallocHost((void **) &_Vm, vectorSize * sizeof(_Vm[0]))); 	//Define pinned memories
	CUDA_SAFE_CALL(cudaMallocHost((void **) &_Cm, vectorSize * sizeof(_Cm[0]))); 	//Define pinned memories
	CUDA_SAFE_CALL(cudaMallocHost((void **) &_Em, vectorSize * sizeof(_Em[0]))); 	//Define pinned memories
	CUDA_SAFE_CALL(cudaMallocHost((void **) &_Rm, vectorSize * sizeof(_Rm[0]))); 	//Define pinned memories

	//Define memory for device
	CUDA_SAFE_CALL(cudaMalloc((void **) &_hm_dev, modelSize*nModel * sizeof(_hm_dev[0])));
	CUDA_SAFE_CALL(cudaMalloc((void **) &_rhs_dev, vectorSize * sizeof(_rhs_dev[0])));
	CUDA_SAFE_CALL(cudaMalloc((void **) &_Vm_dev, vectorSize * sizeof(_Vm_dev[0])));
	CUDA_SAFE_CALL(cudaMalloc((void **) &_Cm_dev, vectorSize * sizeof(_Cm_dev[0])));
	CUDA_SAFE_CALL(cudaMalloc((void **) &_Em_dev, vectorSize * sizeof(_Em_dev[0])));
	CUDA_SAFE_CALL(cudaMalloc((void **) &_Rm_dev, vectorSize * sizeof(_Rm_dev[0])));

	//Fill Vectors
	for(int i=0; i< vectorSize;i++ )
	{
		_Vm[i] = analyzer.allCompartments[i]->initVm;
		_Cm[i] = analyzer.allCompartments[i]->Cm;
		_Em[i] = analyzer.allCompartments[i]->Em;
		_Rm[i] = analyzer.allCompartments[i]->Rm;
	}

	//making Hines Matrices
	for(int i=0; i< nModel;i++ )
	{
		makeHinesMatrix(&network[i], &_hm[i*modelSize]);
//		_printMatrix_Column(nComp,nComp, &_hm[i*modelSize]);

		//TODO: Copy matrix complenetely
		//Copy Matrices
		stat = cublasSetMatrix(nComp, nComp, sizeof(_hm[0]),
						&_hm[i*modelSize], nComp,
						&_hm_dev[i*modelSize], nComp);

		if (stat != CUBLAS_STATUS_SUCCESS) {
				cerr << "CUBLAS setting Variables\n";
				return CuBLASError;
		}
	}

	//Copy to GPU
	sendVector<T>(vectorSize, _Vm,_Vm_dev);
	sendVector<T>(vectorSize, _Em,_Em_dev);
	sendVector<T>(vectorSize, _Cm,_Cm_dev);
	sendVector<T>(vectorSize, _Rm,_Rm_dev);

	//Create Cublas
	if ( cublasCreate(&_handle) != CUBLAS_STATUS_SUCCESS)
	{
		cerr << "CUBLAS initialization failed\n";
		return CuBLASError;
	}

//	thrust::raw_pointer_cast(dev_ptr);

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
//	for (int var = 0; var < nModel*nComp; ++var) {
//		_rhs[var] = var;
////		B[ i ] =V[ i ] * tree[ i ].Cm / ( dt / 2.0 ) +Em[ i ] / tree[ i ].Rm;
//	}
//	_printVector(nModel*nComp, _rhs);
//	cublasSetVector(nModel*nComp, sizeof(_rhs[0]),_rhs, 1,_rhs_dev, 1);
//
//    int ret = dsolve_batch (_hm_dev, _rhs_dev, _Vm_dev, nComp, nModel);
//    assert(!ret);
//
//	cublasGetVector(nModel*nComp, sizeof(_Vm[0]),_Vm_dev, 1,_Vm, 1);
//	//_printVector(nModel*nComp, _Vm);

	return NO_ERROR;
}

template <typename T, int arch>
hscError HSC_SolverComps<T,arch>::UpdateMatrix()
{
	//Copy to GPU
	sendVector<T>(vectorSize, _Vm,_Vm_dev);
	sendVector<T>(vectorSize, _Em,_Em_dev);
	sendVector<T>(vectorSize, _Cm,_Cm_dev);
	sendVector<T>(vectorSize, _Rm,_Rm_dev);

//	cublasSscal(cublasHandle, )

//	thrust::host_vector<int> h_vec(100);
//	thrust::host_vector<int> g_vec(100);
//	for (int i = 0; i < 100; ++i) {
//		h_vec[i] = 1;
//		g_vec[i] = 2;
//	}
//	// transfer to device and compute sum
//	thrust::device_vector<int> d_vec = h_vec;
//	thrust::device_vector<int> d2_vec = g_vec;
//	int x = thrust::transform(d_vec.begin(), d_vec.end(), d2_vec.begin(), d2_vec.end(), thrust::multiplies<int>());
//	int x = thrust::reduce(d_vec.begin(), d_vec.end(), 0, thrust::multiplies<int>());
//	cout << x <<flush;



	cublasGetVector(nModel*nComp, sizeof(_rhs[0]),_rhs_dev,1,_rhs , 1);
	cublasGetVector(nModel*nComp, sizeof(_Vm[0]) ,_Vm_dev, 1, _Vm , 1);
	cublasGetVector(nModel*nComp, sizeof(_Cm[0]) ,_Cm_dev, 1, _Cm , 1);
	cublasGetVector(nModel*nComp, sizeof(_Em[0]) ,_Em_dev, 1, _Em , 1);
	cublasGetVector(nModel*nComp, sizeof(_Rm[0]) ,_Rm_dev, 1, _Rm , 1);



//	B[ i ] =
//		V * Cm / ( dt / 2.0 ) +	Em/Rm;

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
