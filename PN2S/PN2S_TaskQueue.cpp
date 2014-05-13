///////////////////////////////////////////////////////////
//  PN2S_TaskQueue.cpp
//
//  Created on:      26-Dec-2013 4:19:23 PM
//  Original author: Saeed Shariati
///////////////////////////////////////////////////////////

#include "PN2S_TaskQueue.h"
#include <omp.h>

omp_lock_t _empty_lock;
//omp_lock_t _full_lock;

PN2S_TaskQueue::PN2S_TaskQueue(): _limit(100), _size(0){
	omp_init_lock(&_empty_lock);
//	omp_init_lock(&_full_lock);
	omp_set_lock(&_empty_lock);
//	omp_set_lock(&_full_lock);
}


PN2S_TaskQueue::~PN2S_TaskQueue(){
	omp_destroy_lock(&_empty_lock);
//	omp_destroy_lock(&_full_lock);
}


hscError PN2S_TaskQueue::Add(PN2S_TaskInfo &task){
//	if(_list.size() >= _limit)
//		omp_set_lock(&_full_lock);
	#pragma omp critical
	{
		_list.push_front(task);
		_size++;

		cout<<"Thread = "<< omp_get_thread_num()<<"Limit "<<_limit<<" Add   B"<<task.type<<"  QSize="<<_list.size()<<"size="<<_size<<endl<<flush;
	}
	sleep(1);
//	#pragma omp flush(_list)
//	omp_unset_lock(&_empty_lock);

	return  NO_ERROR;
}


PN2S_TaskInfo* PN2S_TaskQueue::Get(){
//	omp_set_lock(&_empty_lock);
	PN2S_TaskInfo* res= &_list[0];
	_list.pop_back();
//	#pragma omp flush(_list)

	#pragma omp critical
	{
		_size--;
		cout<<"Thread = "<< omp_get_thread_num()<<"Limit "<<_limit<<" Get B"<<res->type<<"  QSize="<<_list.size() <<"size="<<_size<<endl<<flush;
	}
//	omp_unset_lock(&_full_lock);
	return res;

	return NULL;
}

//
///**
// * This is the entry point of Process routine.
// * According to the ID it will find ModelPack corresponding to the model and
// * create a TaskInfo for that.
// * Also Read data from Shell for all Models at ModelPack and at the end add the
// * TaskInfo to _scheduleList
// *
// * If Maganer is not started processing, Start it.
// */
//void PN2S_Scheduler::Process(){
//
//}
//
//
//hscError PN2S_Scheduler::getValuesFromShell(){
//
//	return  NO_ERROR;
//}
//
//
//hscError PN2S_Scheduler::AddOutputTask(){
//
//	return  NO_ERROR;
//}
//
//
//PN2S_TaskInfo* PN2S_Scheduler::GetOutputTask(){
//
//	return  NULL;
//}
