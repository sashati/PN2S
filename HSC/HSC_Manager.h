///////////////////////////////////////////////////////////
//  HSC_Manager.h
//  Implementation of the Class HSC_Manager
//  Created on:      26-Dec-2013 4:18:17 PM
//  Original author: Saeed Shariati
///////////////////////////////////////////////////////////

#if !defined(A1182FDE9_1428_42fc_B1C4_EB304C128113__INCLUDED_)
#define A1182FDE9_1428_42fc_B1C4_EB304C128113__INCLUDED_

#include "Definitions.h"
#include "modelSolver/HSCModel.h"
//#include "HSC_Solver.h"
#include "HSC_Scheduler.h"

/**
 *
 */
class HSC_Manager
{
public:
	HSC_Manager();
	virtual ~HSC_Manager();

	hscError Setup();
	hscError InsertModel(HSCModel &model);
	hscError Reinit();
	hscError PrepareSolver();
//	hscError Process();

private:
	vector<HSCModel> _models;
	HSC_Scheduler _scheduler;
//	void startDeviceThreads();

};
#endif // !defined(A1182FDE9_1428_42fc_B1C4_EB304C128113__INCLUDED_)
