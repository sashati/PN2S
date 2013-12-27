///////////////////////////////////////////////////////////
//  HSC_TaskInfo.h
//  Implementation of the Class HSC_TaskInfo
//  Created on:      26-Dec-2013 4:20:33 PM
//  Original author: Saeed Shariati
///////////////////////////////////////////////////////////

#if !defined(AC031FA40_9DD5_4db4_B7F5_2CE46F915EAB__INCLUDED_)
#define AC031FA40_9DD5_4db4_B7F5_2CE46F915EAB__INCLUDED_

#include "modelSolver/HSC_ModelPack.h"
#include "modelSolver/HSC_Solver.h"

struct HSC_TaskInfo
{
	HSC_ModelPack modelPack;
	HSC_Solver solver;
	int type;
};
#endif // !defined(AC031FA40_9DD5_4db4_B7F5_2CE46F915EAB__INCLUDED_)
