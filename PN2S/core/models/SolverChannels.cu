///////////////////////////////////////////////////////////
//  SolverChannels.cpp
//  Implementation of the Class SolverChannels
//  Created on:      27-Dec-2013 7:57:50 PM
//  Original author: Saeed Shariati
///////////////////////////////////////////////////////////

#include "SolverChannels.h"

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

#define NUMBER_OF_MULTI_PROCESSOR 8

SolverChannels::SolverChannels(): _stream(0)
{
}

SolverChannels::~SolverChannels()
{
}

void SolverChannels::AllocateMemory(models::ModelStatistic& s, cudaStream_t stream)
{
	_m_statistic = s;
	_stream = stream;

	if(_m_statistic.nChannels_all == 0)
		return;

	_state.AllocateMemory(_m_statistic.nChannels_all*3);
	_comptIndex.AllocateMemory(_m_statistic.nChannels_all);
	_channel_base.AllocateMemory(_m_statistic.nChannels_all);
	_channel_currents.AllocateMemory(_m_statistic.nChannels_all);

	int threadSize = min(max((int)_m_statistic.nChannels_all/NUMBER_OF_MULTI_PROCESSOR,16), 32);
	_threads=dim3(2,threadSize, 1);
	_blocks=dim3(max((int)(ceil(_m_statistic.nChannels_all / _threads.y)),1), 1);
}

void SolverChannels::PrepareSolver(PField<TYPE_, ARCH_>*  Vm)
{
	if(_m_statistic.nChannels_all)
	{
		_state.Host2Device_Async(_stream);
		_channel_base.Host2Device_Async(_stream);
		_channel_currents.Host2Device_Async(_stream);
		_comptIndex.Host2Device_Async(_stream);
		_Vm = Vm;

		int threadSize = min(max((int)_m_statistic.nChannels_all/8,16), 32);
		_threads=dim3(2,threadSize, 1);
		_blocks=dim3(max((int)(ceil(_m_statistic.nChannels_all / _threads.y)),1), 1);
	}
}

/**
 * KERNELS
 */
extern __shared__ TYPE_ shmem[];

__global__ void advanceChannels(
		TYPE_* v,
		int* compIndex,
		TYPE_* state,
		pn2s::models::ChannelType* ch,
		pn2s::models::ChannelCurrent* current,
		size_t size, TYPE_ dt)
{
	TYPE_ temp, temp2, A, B;
	TYPE_* data = shmem;
	int idx = blockIdx.x * blockDim.y + threadIdx.y;
	if (idx >= size)
		return;

	int i = threadIdx.y * 2 + threadIdx.x;
	data[i] = 1.0;

	TYPE_ power = ch[idx]._xyz_power[threadIdx.x];
	if ( power > 0.0 )
	{
		int cIdx = compIndex[idx];
		TYPE_ x = v[cIdx];


		temp = ch[idx]._xyz_params[threadIdx.x][PARAMS_MIN];
		temp2 = ch[idx]._xyz_params[threadIdx.x][PARAMS_MAX];
		// Check boundaries
		x = fmax(temp, x);
		x = fmin(temp2, x);

		// Calculate new states
		TYPE_ dx = ( temp2 - temp ) / ch[idx]._xyz_params[threadIdx.x][PARAMS_DIV];

		if ( fabs(ch[idx]._xyz_params[threadIdx.x][PARAMS_A_F]) < SINGULARITY ) {
			temp = 0.0;
			A = 0.0;
		} else {
			temp2 = ch[idx]._xyz_params[threadIdx.x][PARAMS_A_C] + exp( ( x + ch[idx]._xyz_params[threadIdx.x][PARAMS_A_D] ) / ch[idx]._xyz_params[threadIdx.x][PARAMS_A_F] );
			if ( fabs( temp2 ) < SINGULARITY ) {
				temp2 = ch[idx]._xyz_params[threadIdx.x][PARAMS_A_C] + exp( ( x + dx/10.0 + ch[idx]._xyz_params[threadIdx.x][PARAMS_A_D] ) / ch[idx]._xyz_params[threadIdx.x][PARAMS_A_F] );
				temp = ( ch[idx]._xyz_params[threadIdx.x][PARAMS_A_A] + ch[idx]._xyz_params[threadIdx.x][PARAMS_A_B] * (x + dx/10 ) ) / temp2;

				temp2 = ch[idx]._xyz_params[threadIdx.x][PARAMS_A_C] + exp( ( x - dx/10.0 + ch[idx]._xyz_params[threadIdx.x][PARAMS_A_D] ) / ch[idx]._xyz_params[threadIdx.x][PARAMS_A_F] );
				temp += ( ch[idx]._xyz_params[threadIdx.x][PARAMS_A_A] + ch[idx]._xyz_params[threadIdx.x][1] * (x - dx/10 ) ) / temp2;
				temp /= 2.0;

				A = temp;
			} else {
				temp = ( ch[idx]._xyz_params[threadIdx.x][PARAMS_A_A] + ch[idx]._xyz_params[threadIdx.x][PARAMS_A_B] * x) / temp2;
				A = temp;
			}
		}

		if ( fabs( ch[idx]._xyz_params[threadIdx.x][9] ) < SINGULARITY ) {
			B = 0.0;
		} else {
			temp2 = ch[idx]._xyz_params[threadIdx.x][7] + exp( ( x + ch[idx]._xyz_params[threadIdx.x][8] ) / ch[idx]._xyz_params[threadIdx.x][9] );
			if ( fabs( temp2 ) < SINGULARITY ) {
				temp2 = ch[idx]._xyz_params[threadIdx.x][7] + exp( ( x + dx/10.0 + ch[idx]._xyz_params[threadIdx.x][8] ) / ch[idx]._xyz_params[threadIdx.x][9] );
				temp = (ch[idx]._xyz_params[threadIdx.x][5] + ch[idx]._xyz_params[threadIdx.x][6] * (x + dx/10) ) / temp2;
				temp2 = ch[idx]._xyz_params[threadIdx.x][7] + exp( ( x - dx/10.0 + ch[idx]._xyz_params[threadIdx.x][8] ) / ch[idx]._xyz_params[threadIdx.x][9] );
				temp += (ch[idx]._xyz_params[threadIdx.x][5] + ch[idx]._xyz_params[threadIdx.x][6] * (x - dx/10) ) / temp2;
				temp /= 2.0;
				B = temp;
			} else {
				B = (ch[idx]._xyz_params[threadIdx.x][5] + ch[idx]._xyz_params[threadIdx.x][6] * x ) / temp2;
			}
		}

		if ( fabs( temp2 ) > SINGULARITY )
			B += temp;

		temp2 = state[3*idx+threadIdx.x];
		if ( ch[idx]._instant& INSTANT_X )
			state[3*idx+threadIdx.x] = A / B;
		else
		{
			temp = 1.0 + dt / 2.0 * B; //new value for temp
			state[3*idx+threadIdx.x] = ( temp2 * ( 2.0 - temp ) + dt * A ) / temp;
		}

		//Update channels characteristics
		data[i] = temp2;
		if (power > 1)
		{
			data[i] *= temp2;
			if (power > 2)
			{
				data[i] *= temp2;
				if (power > 3)
				{
					data[i] *= temp2;
					if (power > 4)
					{
						data[i] = pow( temp2, power);
					}
				}
			}
		}
		__syncthreads();
		if(!threadIdx.x)
			current[idx]._gk = ch[idx]._gbar * data[i] * data[i+1];
	}
}

void SolverChannels::Input()
{

}

void SolverChannels::Process()
{
	if(_m_statistic.nChannels_all < 1)
		return;
	int smem_size = (sizeof(TYPE_) * _threads.x * _threads.y);
//	_Vm->Device2Host();
//	_Vm->print();
//	_state.Device2Host();
//	_state.print();

	advanceChannels <<<_blocks, _threads,smem_size, _stream>>> (
			_Vm->device,
			_comptIndex.device,
			_state.device,
			_channel_base.device,
			_channel_currents.device,
			_m_statistic.nChannels_all,
			_m_statistic.dt);

//	_state.Device2Host();
//	_state.print();
//
//	_channel_currents.Device2Host();
//	_channel_currents.print();
	assert(cudaSuccess == cudaGetLastError());
}

void SolverChannels::Output()
{
	_channel_currents.Device2Host_Async(_stream);
}

/**
 * Set/Get methods
 */

void SolverChannels::SetGateXParams(int index, vector<double>& params)
{
	for (int i = 0; i < min((int)params.size(),13); ++i)
		_channel_base[index]._xyz_params[0][i] = (TYPE_)params[i];
}
void SolverChannels::SetGateYParams(int index, vector<double>& params)
{
	for (int i = 0; i < min((int)params.size(),13); ++i)
		_channel_base[index]._xyz_params[1][i] = (TYPE_)params[i];
}
void SolverChannels::SetGateZParams(int index, vector<double>& params)
{
	for (int i = 0; i < min((int)params.size(),13); ++i)
		_channel_base[index]._xyz_params[2][i] = (TYPE_)params[i];
}

void SolverChannels::SetValue(int index, FIELD::TYPE field, TYPE_ value)
{
	switch(field)
	{
		case FIELD::CH_GBAR:
			_channel_base[index]._gbar = value;
			break;
		case FIELD::CH_GK:
			_channel_currents[index]._gk = value;
			break;
		case FIELD::CH_EK:
			_channel_currents[index]._ek = value;
			break;
		case FIELD::CH_X_POWER:
			_channel_base[index]._xyz_power[0] = (unsigned char)value;
			break;
		case FIELD::CH_Y_POWER:
			_channel_base[index]._xyz_power[1] = (unsigned char)value;
			break;
		case FIELD::CH_Z_POWER:
			_channel_base[index]._xyz_power[2] = (unsigned char)value;
			break;
		case FIELD::CH_X:
			_state[3*index] = value;
			break;
		case FIELD::CH_Y:
			_state[3*index+1] = value;
			break;
		case FIELD::CH_Z:
			_state[3*index+2] = value;
			break;
		case FIELD::CH_COMPONENT_INDEX:
			_comptIndex[index] = (int)value;
			break;
	}
}

TYPE_ SolverChannels::GetValue(int index, FIELD::TYPE field)
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
