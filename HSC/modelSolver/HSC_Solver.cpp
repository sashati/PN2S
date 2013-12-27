///////////////////////////////////////////////////////////
//  HSC_Solver.cpp
//  Implementation of the Class HSC_Solver
//  Created on:      26-Dec-2013 4:21:51 PM
//  Original author: Saeed Shariati
///////////////////////////////////////////////////////////

#include "HSC_Solver.h"

hscError HSC_Solver::Setup(){

	return  NO_ERROR;
}


hscError HSC_Solver::CreateModelPack(vector<HSCModel> models){
	for (vector<HSCModel>::iterator i = models.begin(); i != models.end(); ++i) {

	}

	return NO_ERROR;
}


hscError HSC_Solver::SendDataToGPU(){

	return  NO_ERROR;
}


/**
 * <ul>
 * 	<li>H2D stream call</li>
 * 	<li>Kernel call</li>
 * 	<li>D2H stram call</li>
 * </ul>
 *
 * When the output is ready, send it to the output task list
 */
//void HSC_Solver::DoProcess(HSC_Device* d){
//
//}

//#ifdef DO_UNIT_TESTS
#include <cassert>

using namespace std;

void testHSC_Solver()
{
	cout << "testHSC_Solver" << endl <<flush;

}
//#endif
