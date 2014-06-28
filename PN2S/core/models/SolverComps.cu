///////////////////////////////////////////////////////////
//  SolverComps.cpp
//  Implementation of the Class SolverComps
//  Created on:      27-Dec-2013 7:57:50 PM
//  Original author: Saeed Shariati
///////////////////////////////////////////////////////////

#include "SolverComps.h"
#include "solve.h"

#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

using namespace pn2s::models;
//CuBLAS variables
//cublasHandle_t _handle;

cudaStream_t _stream;

SolverComps::SolverComps(): _models(0), _stream(0)
{
}

SolverComps::~SolverComps()
{
}


Error_PN2S SolverComps::AllocateMemory(models::Model * m, models::ModelStatistic& s, cudaStream_t stream)
{
	_stat = s;
	_models = m;
	_stream = stream;

	if(_stat.nCompts == 0)
		return Error_PN2S::NO_ERROR;

	size_t modelSize = s.nCompts*s.nCompts;
	size_t vectorSize = s.nModels * s.nCompts;

	_hm.AllocateMemory(modelSize*s.nModels);
	_rhs.AllocateMemory(vectorSize);
	_Vm.AllocateMemory(vectorSize);
	_VMid.AllocateMemory(vectorSize);
	_Cm.AllocateMemory(vectorSize);
	_Em.AllocateMemory(vectorSize);
	_Rm.AllocateMemory(vectorSize);
	_Ra.AllocateMemory(vectorSize);

	return Error_PN2S::NO_ERROR;
}
Error_PN2S SolverComps::PrepareSolver()
{
	if(_stat.nCompts == 0)
		return Error_PN2S::NO_ERROR;

	//TODO: OpenMP
	//making Hines Matrices
	for(int i=0; i< _stat.nModels;i++ )
		makeHinesMatrix(&_models[i], &_hm[i*_stat.nCompts*_stat.nCompts]);

	//Copy to GPU
	_hm.Host2Device_Async(_stream);
	_Em.Host2Device_Async(_stream);

//	//Create Cublas
//	if ( cublasCreate(&_handle) != CUBLAS_STATUS_SUCCESS)
//	{
//		return Error_PN2S(Error_PN2S::CuBLASError,
//				"CUBLAS initialization failed");
//	}

	cudaDeviceSynchronize();
	return Error_PN2S::NO_ERROR;
}

/**
 * 			UPDATE MATRIX
 *
 * RHS = Vm * Cm / ( dt / 2.0 ) + Em/Rm;
 *
 */

__global__ void update_rhs(TYPE_* rhs, TYPE_* vm, TYPE_* cm, TYPE_* rm, size_t size, TYPE_ dt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    	rhs[idx] = vm[idx] * cm[idx] / dt * 2.0 + rhs[idx] / rm[idx];
}


__global__ void update_vm(TYPE_* vm, TYPE_* vmid, size_t size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    	vm[idx] = 2.0 * vmid[idx]- vm[idx];
}

void SolverComps::Input()
{
	//Copy to GPU
	_rhs.Send2Device_Async(_Em,_stream); // Em -> rhs
	_Rm.Host2Device_Async(_stream);
	_Vm.Host2Device_Async(_stream);
	_Cm.Host2Device_Async(_stream);
}

void SolverComps::Process()
{
	uint vectorSize = _stat.nModels * _stat.nCompts;

	dim3 threads, blocks;
	threads=dim3(min((vectorSize&0xFFFFFFC0)|0x20,256), 1); //TODO: Check
	blocks=dim3(max(vectorSize / threads.x,1), 1);

	update_rhs <<<blocks, threads,0, _stream>>> (
			_rhs.device,
			_Vm.device,
			_Cm.device,
			_Rm.device,
			vectorSize,
			_stat.dt);
	assert(cudaSuccess == cudaGetLastError());

//	cudaStreamSynchronize(_stream);

	assert(!dsolve_batch (_hm.device, _rhs.device, _VMid.device, _stat.nCompts, _stat.nModels, _stream));

	update_vm <<<blocks, threads,0, _stream>>> (
				_Vm.device,
				_VMid.device,
				vectorSize);

	assert(cudaSuccess == cudaGetLastError());
//	cudaStreamSynchronize(_stream);
}


void SolverComps::Output()
{
	_Vm.Device2Host_Async(_stream);
	cudaStreamSynchronize(_stream);
}


void SolverComps::makeHinesMatrix(models::Model *model, TYPE_ * matrix)
{
	/*
	 * Some convenience variables
	 */
	vector< double > CmByDt(_stat.nCompts);
	vector< double > Ga(_stat.nCompts);
	for ( unsigned int i = 0; i < _stat.nCompts; i++ ) {
		TYPE_ cm = GetValue(model->compts[ i ].location.index, FIELD::CM);
		TYPE_ ra = GetValue(model->compts[ i ].location.index, FIELD::RA);

		CmByDt[i] = cm / ( _stat.dt / 2.0 ) ;
		Ga[i] =  2.0 / ra ;
	}

	/* Each entry in 'coupled' is a list of electrically coupled compartments.
	 * These compartments could be linked at junctions, or even in linear segments
	 * of the cell.
	 */
	vector< vector< unsigned int > > coupled;
	for ( unsigned int i = 0; i < _stat.nCompts; i++ )
		if ( model->compts[ i ].children.size() >= 1 ) {
			coupled.push_back( model->compts[ i ].children );
			coupled.back().push_back( i );
		}

	// Setting diagonal elements
	for ( unsigned int i = 0; i < _stat.nCompts; i++ )
	{
		TYPE_ rm = GetValue(model->compts[ i ].location.index, FIELD::RM);
		matrix[ i * _stat.nCompts + i ] = (TYPE_)(CmByDt[ i ] + 1.0 / rm);
	}


	double gi;
	vector< vector< unsigned int > >::iterator group;
	vector< unsigned int >::iterator ic;
	for ( group = coupled.begin(); group != coupled.end(); ++group ) {
		double gsum = 0.0;

		for ( ic = group->begin(); ic != group->end(); ++ic )
			gsum += Ga[ *ic ];

		for ( ic = group->begin(); ic != group->end(); ++ic ) {
			gi = Ga[ *ic ];

			matrix[ *ic * _stat.nCompts + *ic ] += (TYPE_) (gi * ( 1.0 - gi / gsum ));
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

				matrix[ *ic * _stat.nCompts + *jc ] = (TYPE_)(-gij);
				matrix[ *jc * _stat.nCompts + *ic ] = (TYPE_)(-gij);
			}
		}
	}
//	_printVector(_stat.nCompts*_stat.nCompts,matrix);
}

void SolverComps::SetValue(int index, FIELD::TYPE field, TYPE_ value)
{
	switch(field)
	{
		case FIELD::CM:
			_Cm[index] = value;
			break;
		case FIELD::EM:
			_Em[index] = value;
			break;
		case FIELD::RM:
			_Rm[index] = value;
			break;
		case FIELD::RA:
			_Ra[index] = value;
			break;
		case FIELD::VM:
			_Vm[index] = value;
			break;
		case FIELD::INIT_VM:
			_Vm[index] = value;
			break;
	}
}

TYPE_ SolverComps::GetValue(int index, FIELD::TYPE field)
{
	switch(field)
	{
		case FIELD::CM:
			return _Cm[index];
		case FIELD::EM:
			return _Em[index];
		case FIELD::RM:
			return _Rm[index];
		case FIELD::RA:
			return _Ra[index];
		case FIELD::VM:
			return _Vm[index];
		case FIELD::INIT_VM:
			return _Vm[index];
	}
}

//TYPE_ (*SolverComps::Fetch_Func) (uint, SolverComps::Fields);
