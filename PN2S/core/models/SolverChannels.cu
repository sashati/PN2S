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
	_channels.AllocateMemory(_m_statistic.nChannels_all);
	_channel_currents.AllocateMemory(_m_statistic.nChannels_all);
}

void SolverChannels::PrepareSolver(PField<TYPE_, ARCH_>*  Vm)
{
	_Vm = Vm;
	if(_m_statistic.nChannels_all)
	{
		_state.Host2Device();
		_channels.Host2Device();
		_channel_currents.Host2Device();
	}
}

/**
 * KERNELS
 */
__global__ void advanceChannels(
		TYPE_* v,
		TYPE_* state,
		pn2s::models::ChannelType* ch,
		pn2s::models::ChannelCurrent* current,
		size_t size, TYPE_ dt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size){
		TYPE_ temp, temp2, A, B;
    	TYPE_ x = v[idx];

    	TYPE_ fraction = 1.0;
    	//TODO: distibute this loop and consider every gate with a separate thread
    	for (int xyz = 0; xyz < 2; ++xyz) {
			if ( ch[idx]._xyz_power[xyz] > 0.0 )
			{
				// Check boundaries
				if ( x < ch[idx]._xyz_params[xyz][PARAMS_MIN] )
					x = ch[idx]._xyz_params[xyz][PARAMS_MIN];
				else if ( x > ch[idx]._xyz_params[xyz][PARAMS_MAX] )
					x = ch[idx]._xyz_params[xyz][PARAMS_MAX];

				// Calculate new states
				TYPE_ dx = ( ch[idx]._xyz_params[xyz][PARAMS_MAX] - ch[idx]._xyz_params[xyz][PARAMS_MIN] ) / ch[idx]._xyz_params[xyz][PARAMS_DIV];
				if ( fabs(ch[idx]._xyz_params[xyz][PARAMS_A_F]) < SINGULARITY ) {
					temp = 0.0;
					A = 0.0;
				} else {
					temp2 = ch[idx]._xyz_params[xyz][PARAMS_A_C] + exp( ( x + ch[idx]._xyz_params[xyz][PARAMS_A_D] ) / ch[idx]._xyz_params[xyz][PARAMS_A_F] );
					if ( fabs( temp2 ) < SINGULARITY ) {
						temp2 = ch[idx]._xyz_params[xyz][PARAMS_A_C] + exp( ( x + dx/10.0 + ch[idx]._xyz_params[xyz][PARAMS_A_D] ) / ch[idx]._xyz_params[xyz][PARAMS_A_F] );
						temp = ( ch[idx]._xyz_params[xyz][PARAMS_A_A] + ch[idx]._xyz_params[xyz][PARAMS_A_B] * (x + dx/10 ) ) / temp2;

						temp2 = ch[idx]._xyz_params[xyz][PARAMS_A_C] + exp( ( x - dx/10.0 + ch[idx]._xyz_params[xyz][PARAMS_A_D] ) / ch[idx]._xyz_params[xyz][PARAMS_A_F] );
						temp += ( ch[idx]._xyz_params[xyz][PARAMS_A_A] + ch[idx]._xyz_params[xyz][1] * (x - dx/10 ) ) / temp2;
						temp /= 2.0;

						A = temp;
					} else {
						temp = ( ch[idx]._xyz_params[xyz][PARAMS_A_A] + ch[idx]._xyz_params[xyz][PARAMS_A_B] * x) / temp2;
						A = temp;
					}
				}
				if ( fabs( ch[idx]._xyz_params[xyz][9] ) < SINGULARITY ) {
					B = 0.0;
				} else {
					temp2 = ch[idx]._xyz_params[xyz][7] + exp( ( x + ch[idx]._xyz_params[xyz][8] ) / ch[idx]._xyz_params[xyz][9] );
					if ( fabs( temp2 ) < SINGULARITY ) {
						temp2 = ch[idx]._xyz_params[xyz][7] + exp( ( x + dx/10.0 + ch[idx]._xyz_params[xyz][8] ) / ch[idx]._xyz_params[xyz][9] );
						temp = (ch[idx]._xyz_params[xyz][5] + ch[idx]._xyz_params[xyz][6] * (x + dx/10) ) / temp2;
						temp2 = ch[idx]._xyz_params[xyz][7] + exp( ( x - dx/10.0 + ch[idx]._xyz_params[xyz][8] ) / ch[idx]._xyz_params[xyz][9] );
						temp += (ch[idx]._xyz_params[xyz][5] + ch[idx]._xyz_params[xyz][6] * (x - dx/10) ) / temp2;
						temp /= 2.0;
						B = temp;
					} else {
						B = (ch[idx]._xyz_params[xyz][5] + ch[idx]._xyz_params[xyz][6] * x ) / temp2;
					}
				}
				if ( fabs( temp2 ) > SINGULARITY )
					B += temp;

				temp2 = state[3*idx+xyz];
				if ( ch[idx]._instant& INSTANT_X )
					state[3*idx+xyz] = A / B;
				else
				{
					temp = 1.0 + dt / 2.0 * B; //new value for temp
					state[3*idx+xyz] = ( temp2 * ( 2.0 - temp ) + dt * A ) / temp;
				}

				//Update channels characteristics
				switch(ch[idx]._xyz_power[xyz])
				{
					case 1:
						fraction = fraction * temp2;
						break;
					case 2:
						fraction = fraction * temp2 * temp2;
						break;
					case 3:
						fraction = fraction * temp2 * temp2 * temp2;
						break;
					case 4:
						fraction = fraction * temp2 * temp2 * temp2 * temp2;
						break;
					default:
						fraction = fraction * pow( temp2, (TYPE_)ch[idx]._xyz_power[xyz]);
						break;
				}
			}
    	}
    	current[idx]._gk = ch[idx]._gbar * fraction;
    }
}

void SolverChannels::Input()
{

}

void SolverChannels::Process()
{
	dim3 threads, blocks;
	threads=dim3(min((int)(_m_statistic.nChannels_all&0xFFFFFFC0)|0x20,256), 1);
	blocks=dim3(max((int)(_m_statistic.nChannels_all / threads.x),1), 1);

	_channels.print();
	advanceChannels <<<blocks, threads,0, _stream>>> (
			_Vm->device,
			_state.device,
			_channels.device,
			_channel_currents.device,
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
		_channels[index]._xyz_params[0][i] = (TYPE_)params[i];
}
void SolverChannels::SetGateYParams(int index, vector<double> params)
{
	for (int i = 0; i < min((int)params.size(),13); ++i)
		_channels[index]._xyz_params[1][i] = (TYPE_)params[i];
}
void SolverChannels::SetGateZParams(int index, vector<double> params)
{
	for (int i = 0; i < min((int)params.size(),13); ++i)
		_channels[index]._xyz_params[2][i] = (TYPE_)params[i];
}

void SolverChannels::SetValue(int index, FIELD::TYPE field, TYPE_ value)
{
	switch(field)
	{
		case FIELD::CH_GBAR:
			_channels[index]._gbar = value;
			break;
		case FIELD::CH_GK:
			_channel_currents[index]._gk = value;
			break;
		case FIELD::CH_EK:
			_channel_currents[index]._ek = value;
			break;
		case FIELD::CH_X_POWER:
			_channels[index]._xyz_power[0] = (unsigned char)value;
			break;
		case FIELD::CH_Y_POWER:
			_channels[index]._xyz_power[1] = (unsigned char)value;
			break;
		case FIELD::CH_Z_POWER:
			_channels[index]._xyz_power[2] = (unsigned char)value;
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
