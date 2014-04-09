///////////////////////////////////////////////////////////
//  HSC_OutputManager.h
//  Implementation of the Class HSC_OutputManager
//  Created on:      26-Dec-2013 4:18:18 PM
//  Original author: Saeed Shariati
///////////////////////////////////////////////////////////

#if !defined(EA_7219F77E_D358_4d9b_A477_5363274F9118__INCLUDED_)
#define EA_7219F77E_D358_4d9b_A477_5363274F9118__INCLUDED_

#include "Definitions.h"
#include "HSC_TaskInfo.h"
#include "HSC_OutputWriter.h"
#include "HSC_Scheduler.h"

/**
 * This class is responsible to get output tasks from scheduler and write them to
 * output. For example send the result to the Shell.
 * It contains a thread that is wait on output list at Scheduler 
 */
class HSC_OutputManager
{

public:
	HSC_OutputManager();
	virtual ~HSC_OutputManager();
	HSC_OutputWriter *m_HSC_OutputWriter;
	HSC_Scheduler *m_HSC_Scheduler;

//	hscError_t SendDataToShell(HSC_TaskInfo task);

};
#endif // !defined(EA_7219F77E_D358_4d9b_A477_5363274F9118__INCLUDED_)
