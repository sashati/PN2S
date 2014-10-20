///////////////////////////////////////////////////////////
//  SolverGates.cpp
//  Implementation of the Class SolverGates
//  Created on:      27-Dec-2013 7:57:50 PM
//  Original author: Saeed Shariati
///////////////////////////////////////////////////////////

#include "SolverGates.h"

#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <math.h>

using namespace pn2s::models;

#define SINGULARITY 1.0e-6

//A mask to check INSTANT variable in the channel
#define INSTANT_X 1
#define INSTANT_Y 2
#define INSTANT_Z 4
#define IS_SECOND_GATE 8

#define NUMBER_OF_MULTI_PROCESSOR 8

#define PARAM_SIZE	13

SolverGates::SolverGates(): _stream(0), _Vm(0)
{
}

SolverGates::~SolverGates()
{
}

size_t SolverGates::AllocateMemory(models::ModelStatistic& s, cudaStream_t stream)
{
	_m_statistic = s;
	_stream = stream;

	if(_m_statistic.nGates <= 0)
		return 0;

	_nCompt = s.nCompts_per_model;
	size_t val = 0;
	val += _ch_currents_gk.AllocateMemory(_m_statistic.nChannels_all);
	val += _ch_currents_gkek.AllocateMemory(_m_statistic.nChannels_all);

	val += _state.AllocateMemory(_m_statistic.nGates, 0);
	val += _gk.AllocateMemory(_m_statistic.nChannels_all, 0); //Channel currents

	//Indices
	val += _comptIndex.AllocateMemory(_m_statistic.nGates, 0);
	val += _channelIndex.AllocateMemory(_m_statistic.nGates, 0);
	val += _gateIndex.AllocateMemory(_m_statistic.nGates, 0);

	//Constant values
	val += _ek.AllocateMemory(_m_statistic.nGates, 0);
	val += _gbar.AllocateMemory(_m_statistic.nGates, 0);
	val += _power.AllocateMemory(_m_statistic.nGates, 0);
	val += _params.AllocateMemory(_m_statistic.nGates, 0);
	val += _params_div_min_max.AllocateMemory(_m_statistic.nGates, 0);

	int threadSize = min(max((int)_m_statistic.nChannels_all/8,16), 32);
	_threads=dim3(2,threadSize, 1);
	_blocks=dim3(max((int)(ceil((double)_m_statistic.nChannels_all / _threads.y)),1), 1);

	return val;
}

void SolverGates::PrepareSolver(PField<TYPE_>*  Vm, PField<TYPE_>*  hm,  PField<TYPE_>*  rhs)
{
	if(_m_statistic.nGates)
	{
		_ch_currents_gk.Host2Device_Async(_stream);
		_ch_currents_gkek.Host2Device_Async(_stream);

		_state.Host2Device_Async(_stream);
		_gk.Host2Device_Async(_stream);
		_comptIndex.Host2Device_Async(_stream);
		_channelIndex.Host2Device_Async(_stream);
		_gateIndex.Host2Device_Async(_stream);
		_ek.Host2Device_Async(_stream);
		_gbar.Host2Device_Async(_stream);
		_power.Host2Device_Async(_stream);
		_params.Host2Device_Async(_stream);
		_params_div_min_max.Host2Device_Async(_stream);

		_Vm = Vm;
		_rhs = rhs;
		_hm = hm;

		_threads=dim3(32*3);
		_blocks=dim3(ceil(_m_statistic.nGates / (double)_threads.x));
	}
}

/**
 * KERNELS
 */

__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                          (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
            old = atomicCAS(address_as_ull, assumed,__double_as_longlong(val +
                               __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__global__ void advanceGates(
		TYPE_*  state,
		TYPE_*  gk,
		TYPE_* current_gk,
		TYPE_* current_gk_ek,
		TYPE_*  power,
		pn2s::models::GateParams* params,
		TYPE3_* div_min_max,
		TYPE_* gbar,
		TYPE_*  ek,
		int*  comptIndex,
		int*  channelIndex,
		int*  gateIndex,
		TYPE_* v,
		TYPE_* rhs,
		size_t size, TYPE_ dt)
{
	extern __shared__ TYPE2_ data[];
	TYPE_ temp, temp2, A, B;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;

	int fi = gateIndex[idx];

	if ( power[idx] > 0.0 )
	{
		TYPE_ x = v[comptIndex[idx]];

		temp = div_min_max[idx].y;
		temp2 = div_min_max[idx].z;

		// Calculate new states
		TYPE_ dx = ( temp2 - temp ) / div_min_max[idx].x;

		// Check boundaries
		x = fmax(temp, x);
		x = fmin(temp2, x);


		if ( fabs(params[idx].p[PARAMS_A_F]) < SINGULARITY ) {
			temp = 0.0;
			A = 0.0;
		} else {
			temp2 = params[idx].p[PARAMS_A_C] + exp( ( x + params[idx].p[PARAMS_A_D] ) / params[idx].p[PARAMS_A_F] );
			if ( fabs( temp2 ) < SINGULARITY ) {
				temp2 = params[idx].p[PARAMS_A_C] + exp( ( x + dx/10.0 + params[idx].p[PARAMS_A_D] ) / params[idx].p[PARAMS_A_F] );
				temp = ( params[idx].p[PARAMS_A_A] + params[idx].p[PARAMS_A_B] * (x + dx/10 ) ) / temp2;

				temp2 = params[idx].p[PARAMS_A_C] + exp( ( x - dx/10.0 + params[idx].p[PARAMS_A_D] ) / params[idx].p[PARAMS_A_F] );
				temp += ( params[idx].p[PARAMS_A_A] + params[idx].p[1] * (x - dx/10 ) ) / temp2;
				temp /= 2.0;

				A = temp;
			} else {
				temp = ( params[idx].p[PARAMS_A_A] + params[idx].p[PARAMS_A_B] * x) / temp2;
				A = temp;
			}
		}

		if ( fabs( params[idx].p[PARAMS_B_F] ) < SINGULARITY ) {
			B = 0.0;
		} else {
			temp2 = params[idx].p[7] + exp( ( x + params[idx].p[8] ) / params[idx].p[9] );
			if ( fabs( temp2 ) < SINGULARITY ) {
				temp2 = params[idx].p[7] + exp( ( x + dx/10.0 + params[idx].p[8] ) / params[idx].p[9] );
				temp = (params[idx].p[5] + params[idx].p[6] * (x + dx/10) ) / temp2;
				temp2 = params[idx].p[7] + exp( ( x - dx/10.0 + params[idx].p[8] ) / params[idx].p[9] );
				temp += (params[idx].p[5] + params[idx].p[6] * (x - dx/10) ) / temp2;
				temp /= 2.0;
				B = temp;
			} else {
				B = (params[idx].p[5] + params[idx].p[6] * x ) / temp2;
			}
		}

		if ( fabs( temp2 ) > SINGULARITY )
			B += temp;

		temp2 = state[idx];
		temp = 1.0 + dt / 2.0 * B; //new value for temp
		state[idx] = ( temp2 * ( 2.0 - temp ) + dt * A ) / temp;

		__syncthreads();
		//Update channels characteristics
		data[threadIdx.x].x = temp2;
		if (power[idx] > 1)
		{
			data[threadIdx.x].x *= temp2;
			if (power[idx] > 2)
			{
				data[threadIdx.x].x *= temp2;
				if (power[idx] > 3)
				{
					data[threadIdx.x].x *= temp2;
					if (power[idx] > 4)
					{
						data[threadIdx.x].x = pow( temp2, power[idx]);
					}
				}
			}
		}
		__syncthreads();

		unsigned char bitmap = 1;
		if(fi & bitmap)
		{
			data[threadIdx.x-1].x *= data[threadIdx.x].x;
			data[threadIdx.x].x = 0;
		}
		__syncthreads();
		data[threadIdx.x].x = gbar[idx] *data[threadIdx.x].x;
		data[threadIdx.x].y = ek[idx] *data[threadIdx.x].x;

		if(!(fi & bitmap))
		{
			current_gk[channelIndex[idx]] = data[threadIdx.x].x;

//			int cidx = comptIndex[idx];
//			current_gk[cidx] = 0;
//			current_gk_ek[cidx] = 0;
//			atomicAdd(current_gk + cidx, data[threadIdx.x].x);
//			atomicAdd(current_gk_ek + cidx, data[threadIdx.x].y);
		}
	}
}

double SolverGates::Input()
{
	return 0;
}

double SolverGates::Process()
{
	clock_t	start_time = clock();
	if(_m_statistic.nGates > 0)
	{
		int smem_size = (sizeof(TYPE2_) * _threads.x);
		advanceGates <<<_blocks, _threads, smem_size, _stream>>> (
				_state.device,
				_gk.device,
				_ch_currents_gk.device,
				_ch_currents_gkek.device,
				_power.device,
				_params.device,
				_params_div_min_max.device,
				_gbar.device,
				_ek.device,
				_comptIndex.device,
				_channelIndex.device,
				_gateIndex.device,
				_Vm->device,
				_rhs->device,
				_m_statistic.nGates, _m_statistic.dt);
		assert(cudaSuccess == cudaGetLastError());
	}

	double elapsed_time = ( std::clock() - start_time );
//	cout << "GATE: " << elapsed_time << endl << flush;
	return elapsed_time;
}

double SolverGates::Output()
{
	clock_t	start_time = clock();

//	_ch_currents_gk_ek.Device2Host_Async(_stream);

	return std::clock() - start_time ;
}

/**
 * Set/Get methods
 */

void SolverGates::SetGateParams(int index, vector<double>& params)
{
	for (int i = 0; i < min((int)params.size(),13); ++i)
		_params[index].p[i] = (TYPE_)params[i];

	_params_div_min_max[index].x = (TYPE_)params[PARAMS_DIV];
	_params_div_min_max[index].y = (TYPE_)params[PARAMS_MIN];
	_params_div_min_max[index].z = (TYPE_)params[PARAMS_MAX];
}

void SolverGates::SetValue(int index, FIELD::GATE field, TYPE_ value)
{
	switch(field)
	{
		case FIELD::GATE_CH_GBAR:
			_gbar[index] = value;
			break;
		case FIELD::GATE_CH_GK:
			_gk[index] = value;
			_ch_currents_gk[_channelIndex[index]] = value;
			break;
		case FIELD::GATE_CH_EK:
			_ek[index] = value;
			_ch_currents_gkek[_channelIndex[index]] = value;
			break;
		case FIELD::GATE_POWER:
			_power[index] = (unsigned char)value;
			break;
		case FIELD::GATE_STATE:
			_state[index] = value;
			break;
		case FIELD::GATE_COMPONENT_INDEX:
			_comptIndex[index] = (int)value;
			break;
		case FIELD::GATE_CHANNEL_INDEX:
			_channelIndex[index] = (int)value;
			break;
		case FIELD::GATE_INDEX:
			_gateIndex[index] = (int)value;
			break;
	}
}

TYPE_ SolverGates::GetValue(int index, FIELD::GATE field)
{
//	switch(field)
//	{
//		case FIELD::CH_GBAR:
//			return _gbar[index];
//		case FIELD::CH_X_POWER:
//			return _xPower[index];
//		case FIELD::CH_Y_POWER:
//			return _yPower[index];
//		case FIELD::CH_Z_POWER:
//			return _zPower[index];
//	}
	return 0;
}
