///////////////////////////////////////////////////////////
//  HSC_Scheduler.cpp
//  Implementation of the Class HSC_Scheduler
//  Created on:      26-Dec-2013 4:19:23 PM
//  Original author: Saeed Shariati
///////////////////////////////////////////////////////////

#include "HSC_Scheduler.h"




HSC_Scheduler::HSC_Scheduler(){

}


HSC_Scheduler::~HSC_Scheduler(){

}


hscError HSC_Scheduler::AddInputTask(HSC_TaskInfo &task){
	_inputTasks.push_front(task);
	return  NO_ERROR;
}


HSC_TaskInfo* HSC_Scheduler::GetInputTask(){
	HSC_TaskInfo* res= &_inputTasks[0];
	_inputTasks.pop_back();
	return res;
}


/**
 * This is the entry point of Process routine.
 * According to the ID it will find ModelPack corresponding to the model and
 * create a TaskInfo for that.
 * Also Read data from Shell for all Models at ModelPack and at the end add the
 * TaskInfo to _scheduleList
 * 
 * If Maganer is not started processing, Start it.
 */
void HSC_Scheduler::Process(){

}


hscError HSC_Scheduler::getValuesFromShell(){

	return  NO_ERROR;
}


hscError HSC_Scheduler::AddOutputTask(){

	return  NO_ERROR;
}


HSC_TaskInfo* HSC_Scheduler::GetOutputTask(){

	return  NULL;
}
