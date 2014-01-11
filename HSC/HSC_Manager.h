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
#include "modelSolver/HSC_Solver.h"
#include "HSC_Scheduler.h"
#include "HSC_Device.h"
#include "HSC_DeviceManager.h"

/**
 * This class has a thread to look at scheduler for new tasks and send them to
 * devices.
 */
class HSC_Manager
{
public:
	static hscError Setup(double dt);
	static hscError Reinit();
	static hscError PrepareSolver();
	static hscError AddInputTask(uint id);
	static hscError Process(uint id);

private:
	static void startDeviceThreads();

};
#endif // !defined(A1182FDE9_1428_42fc_B1C4_EB304C128113__INCLUDED_)
