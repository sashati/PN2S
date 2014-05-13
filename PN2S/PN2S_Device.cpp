///////////////////////////////////////////////////////////
//  PN2S_Device.cpp
//  Implementation of the Class PN2S_Device
//  Created on:      26-Dec-2013 4:18:01 PM
//  Original author: saeed
///////////////////////////////////////////////////////////

#include "PN2S_Device.h"
#include "Definitions.h"
#include <cuda.h>
#include <cuda_runtime.h>

PN2S_Device::PN2S_Device(int _id): id(_id){
	cudaDeviceReset();
	_queue_size = 2;

}

PN2S_Device::~PN2S_Device(){
	cudaDeviceReset();
}


hscError PN2S_Device::SelectDevice(){
	cudaDeviceProp deviceProp;
	cudaSetDevice(id);
	//TODO: Set configurations in device
//	cudaGetDeviceProperties(&deviceProp, id);
	return  NO_ERROR;
}

hscError PN2S_Device::PrepareSolver(vector<PN2SModel> &m,  double dt){
	return _solver.PrepareSolver(m,dt);
}

/**
 * Multithread tasks
 */

int _iq_size;
int _iq_limit;
std::deque<PN2S_ModelPack> _iq;

//No limit size for output queue
int _oq_size;
std::deque<PN2S_ModelPack> _oq;

void PN2S_Device::Process()
{
	omp_lock_t _empty_lock_input;
	omp_lock_t _full_lock_input;
	omp_init_lock(&_empty_lock_input);
	omp_init_lock(&_full_lock_input);
	omp_set_lock(&_empty_lock_input);
	omp_set_lock(&_full_lock_input);

	omp_lock_t _empty_lock_output;
	omp_init_lock(&_empty_lock_output);
	omp_set_lock(&_empty_lock_output);

	//Create queues for task scheduling
	_iq_limit = _queue_size;
	#pragma omp parallel \
		shared(_iq, _iq_size) \
		firstprivate(_iq_limit) \
		num_threads(3)
	{
		int tid = omp_get_thread_num();
		if(tid%3 == 0)
		{
			task1_prepareInput(_empty_lock_input,_full_lock_input);
		}
		else if(tid%3==1)
		{
			task2_DoProcess(_empty_lock_input,_full_lock_input,_empty_lock_output);
		}
		else if(tid%3==2)
		{
			task3_prepareOutput(_empty_lock_output);
		}
	}

	omp_destroy_lock(&_empty_lock_input);

}

void PN2S_Device::task1_prepareInput(omp_lock_t& _empty_lock,omp_lock_t& _full_lock) {
	for (vector<PN2S_ModelPack>::iterator it = _solver.modelPacks.begin(); it != _solver.modelPacks.end(); ++it)
	{
		if(_iq_size >= _iq_limit)
			omp_set_lock(&_full_lock);

		sleep(.1);
		#pragma omp critical
		{
			_iq.push_back(*it);
			_iq_size++;

			#pragma omp flush(_iq_size)
			cout<<"Thread = "<< omp_get_thread_num()<<"Limit "<<_iq_limit<<" Add"<<"  QSize="<<_iq.size()<<"size="<<_iq_size<<endl<<flush;
		}
		omp_unset_lock(&_empty_lock);
	}
}

void PN2S_Device::task2_DoProcess(omp_lock_t& _empty_lock_input,
		omp_lock_t& _full_lock_input,
		omp_lock_t& _empty_lock_output)
{
	for (int i = 0; i < 10; i++) {

		if(_iq_size == 0)
			omp_set_lock(&_empty_lock_input);

		PN2S_ModelPack& t= _iq[0];
		_iq.pop_front();

		//Do Process
		sleep(.5);

		//Add task to output
		_oq.push_back(t);

		#pragma omp critical
		{
			_iq_size--;
			_oq_size++;
			cout<<"Thread = "<< omp_get_thread_num()<<"Limit "<<_iq_limit<<" Process "<<"  QSize="<<_iq.size() <<"size="<<_iq_size<<endl<<flush;
		}

		omp_unset_lock(&_full_lock_input);
		omp_unset_lock(&_empty_lock_output);
	}
}

void PN2S_Device::task3_prepareOutput(omp_lock_t& _empty_lock) {
	for (int i = 0; i < 10; i++) {
		if(_oq_size == 0)
			omp_set_lock(&_empty_lock);

		sleep(1);
		PN2S_ModelPack* res= &_oq[0];
		_oq.pop_front();

		//Output routins
		sleep(.5);

		#pragma omp critical
		{
			_oq_size--;
			cout<<"Thread = "<< omp_get_thread_num()<<" Output "<<"  QSize="<<_oq.size() <<"size="<<_oq_size<<endl<<flush;
		}
	}
}

