///////////////////////////////////////////////////////////
//  HSC_Solver.cpp
//  Implementation of the Class HSC_Solver
//  Created on:      26-Dec-2013 4:21:51 PM
//  Original author: Saeed Shariati
///////////////////////////////////////////////////////////

#include "HSC_Solver.h"
#include <assert.h>

HSC_Solver::HSC_Solver()
{
	_dt = 1; //1ms
	solverPacks.clear();
	_modelToPackMap.clear();
}

HSC_Solver::~HSC_Solver()
{

}

hscError HSC_Solver::Setup(double dt){
	_dt = dt;
	return  NO_ERROR;
}


hscError HSC_Solver::PrepareSolver(vector<HSCModel> &m){
	//Estimate size of modelpack
	HSC_SolverData d(_dt);

	solverPacks.push_back(d);
	// Prepare solver for each modelpack
	hscError res = solverPacks[0].PrepareSolver(m);

	//Assign keys to modelPack, to be able to find later
	for(vector<HSCModel>::iterator it = m.begin(); it != m.end(); ++it) {
		_modelToPackMap[it->id] = &solverPacks[0];
	}
	return res;
}


HSC_SolverData* HSC_Solver::LocateDataByID(hscID_t id){
	return _modelToPackMap[id];
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
void HSC_Solver::Process(HSC_SolverData* data, HSC_Device* d){
	hscError res = data->Process();
	assert(!res);
}

//#ifdef DO_UNIT_TESTS
#include <cassert>

using namespace std;

void testHSC_Solver()
{
	cout << "testHSC_Solver" << endl <<flush;

}
//#endif


