///////////////////////////////////////////////////////////
//  SolverChannels.cpp
//  Implementation of the Class SolverChannels
//  Created on:      27-Dec-2013 7:57:50 PM
//  Original author: Saeed Shariati
///////////////////////////////////////////////////////////

#include "SolverChannels.h"
#include <math.h>

#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

using namespace pn2s::models;

#define SINGULARITY 1.0e-6

//A mask to check INSTANT variable in the channel
#define INSTANT_X 1
#define INSTANT_Y 2
#define INSTANT_Z 4

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

//	_gbar.AllocateMemory(_m_statistic.nChannels_all);
	_channels.AllocateMemory(_m_statistic.nChannels_all);

//	_gk.AllocateMemory(_m_statistic.nChannels_all);
//	_ek.AllocateMemory(_m_statistic.nChannels_all);
}

void SolverChannels::PrepareSolver()
{
	if(_m_statistic.nChannels_all)
	{
		//Copy to GPU
//		_gbar.Host2Device();
		_channels.Host2Device();

//		_gk.Host2Device();
//		_ek.Host2Device();
	}
}

/**
 * KERNELS
 */
__global__ void advanceChannels(
		TYPE_* v,
		pn2s::models::ChannelType* ch,
		size_t size, TYPE_ dt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size){
		TYPE_ temp, temp2, A, B;
    	TYPE_ x = v[idx];
    	if ( ch[idx]._x_power > 0.0 )
		{
    		if ( x < ch[idx]._x_params[PARAMS_MIN] )
				x = ch[idx]._x_params[PARAMS_MIN];
			else if ( x > ch[idx]._x_params[PARAMS_MAX] )
				x = ch[idx]._x_params[PARAMS_MAX];

    		TYPE_ dx = ( ch[idx]._x_params[PARAMS_MAX] - ch[idx]._x_params[PARAMS_MIN] ) / ch[idx]._x_params[PARAMS_DIV];
    		if ( fabs(ch[idx]._x_params[PARAMS_A_F]) < SINGULARITY ) {
				temp = 0.0;
				A = 0.0;
			} else {
				temp2 = ch[idx]._x_params[PARAMS_A_C] + exp( ( x + ch[idx]._x_params[PARAMS_A_D] ) / ch[idx]._x_params[PARAMS_A_F] );
				if ( fabs( temp2 ) < SINGULARITY ) {
					temp2 = ch[idx]._x_params[PARAMS_A_C] + exp( ( x + dx/10.0 + ch[idx]._x_params[PARAMS_A_D] ) / ch[idx]._x_params[PARAMS_A_F] );
					temp = ( ch[idx]._x_params[PARAMS_A_A] + ch[idx]._x_params[PARAMS_A_B] * (x + dx/10 ) ) / temp2;

					temp2 = ch[idx]._x_params[PARAMS_A_C] + exp( ( x - dx/10.0 + ch[idx]._x_params[PARAMS_A_D] ) / ch[idx]._x_params[PARAMS_A_F] );
					temp += ( ch[idx]._x_params[PARAMS_A_A] + ch[idx]._x_params[1] * (x - dx/10 ) ) / temp2;
					temp /= 2.0;

					A = temp;
				} else {
					temp = ( ch[idx]._x_params[PARAMS_A_A] + ch[idx]._x_params[PARAMS_A_B] * x) / temp2;
					A = temp;
				}
			}
			if ( fabs( ch[idx]._x_params[9] ) < SINGULARITY ) {
				B = 0.0;
			} else {
				temp2 = ch[idx]._x_params[7] + exp( ( x + ch[idx]._x_params[8] ) / ch[idx]._x_params[9] );
				if ( fabs( temp2 ) < SINGULARITY ) {
					temp2 = ch[idx]._x_params[7] + exp( ( x + dx/10.0 + ch[idx]._x_params[8] ) / ch[idx]._x_params[9] );
					temp = (ch[idx]._x_params[5] + ch[idx]._x_params[6] * (x + dx/10) ) / temp2;
					temp2 = ch[idx]._x_params[7] + exp( ( x - dx/10.0 + ch[idx]._x_params[8] ) / ch[idx]._x_params[9] );
					temp += (ch[idx]._x_params[5] + ch[idx]._x_params[6] * (x - dx/10) ) / temp2;
					temp /= 2.0;
					B = temp;
				} else {
					B = (ch[idx]._x_params[5] + ch[idx]._x_params[6] * x ) / temp2;
				}
			}
			if ( fabs( temp2 ) > SINGULARITY )
				B += temp;

			if ( ch[idx]._instant& INSTANT_X )
				ch[idx]._x = A / B;
			else
			{
				temp = 1.0 + dt / 2.0 * B; //new value for temp
				ch[idx]._x = ( ch[idx]._x * ( 2.0 - temp ) + dt * A ) / temp;
			}
		}
    }
}

void SolverChannels::Input()
{

}

void SolverChannels::Process(PField<TYPE_, ARCH_>*  _Vm)
{
	dim3 threads, blocks;
	threads=dim3(min((int)(_m_statistic.nChannels_all&0xFFFFFFC0)|0x20,256), 1);
	blocks=dim3(max((int)(_m_statistic.nChannels_all / threads.x),1), 1);

	_channels.print();
	advanceChannels <<<blocks, threads,0, _stream>>> (
			_Vm->device,
			_channels.device,
			_m_statistic.nChannels_all,
			_m_statistic.dt);
	assert(cudaSuccess == cudaGetLastError());

	_channels.Device2Host();
	_channels.print();
}


void SolverChannels::Output()
{
//	_Vm.Device2Host_Async(_stream);
//	cudaStreamSynchronize(_stream);
}

void SolverChannels::SetGateXParams(int index, vector<double> params)
{
	for (int i = 0; i < min((int)params.size(),13); ++i)
		_channels[index]._x_params[i] = (TYPE_)params[i];
}
void SolverChannels::SetGateYParams(int index, vector<double> params)
{
	for (int i = 0; i < min((int)params.size(),13); ++i)
		_channels[index]._y_params[i] = (TYPE_)params[i];
}
void SolverChannels::SetGateZParams(int index, vector<double> params)
{
	for (int i = 0; i < min((int)params.size(),13); ++i)
		_channels[index]._z_params[i] = (TYPE_)params[i];
}

void SolverChannels::SetValue(int index, FIELD::TYPE field, TYPE_ value)
{
	switch(field)
	{
//		case FIELD::CH_GBAR:
//			_gbar[index] = value;
//			break;
		case FIELD::CH_X_POWER:
			_channels[index]._x_power = (unsigned char)value;
			break;
		case FIELD::CH_Y_POWER:
			_channels[index]._y_power = (unsigned char)value;
			break;
		case FIELD::CH_Z_POWER:
			_channels[index]._y_power = (unsigned char)value;
			break;
		case FIELD::CH_X:
			_channels[index]._x = value;
			break;
		case FIELD::CH_Y:
			_channels[index]._y = value;
			break;
		case FIELD::CH_Z:
			_channels[index]._z = value;
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
