///////////////////////////////////////////////////////////
//  HSC_Scheduler.h
//  Implementation of the Class HSC_Scheduler
//  Created on:      26-Dec-2013 4:19:23 PM
//  Original author: Saeed Shariati
///////////////////////////////////////////////////////////

#if !defined(AE33FA2BC_2680_47eb_9858_1D1498C6FAE0__INCLUDED_)
#define AE33FA2BC_2680_47eb_9858_1D1498C6FAE0__INCLUDED_

#include "HSC_TaskInfo.h"

class HSC_Scheduler
{

public:
	HSC_TaskInfo *m_HSC_TaskInfo;

	HSC_Scheduler();
	virtual ~HSC_Scheduler();
	hscError AddInputTask(HSC_TaskInfo task);
	HSC_TaskInfo* GetInputTask();
	void Process();
	hscError AddOutputTask();
	HSC_TaskInfo* GetOutputTask();

private:
	hscError getValuesFromShell();

};
#endif // !defined(AE33FA2BC_2680_47eb_9858_1D1498C6FAE0__INCLUDED_)
