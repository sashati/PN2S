///////////////////////////////////////////////////////////
//  HSC_Manager.cpp
//  Implementation of the Class HSC_Manager
//
//  Created on:      26-Dec-2013 4:18:17 PM
//  Original author: Saeed Shariati
///////////////////////////////////////////////////////////

#include "HSC_Manager.h"
#include "modelSolver/HSCModel_Base.h"
#include "modelSolver/HSCModel_HHChannel.h"


HSC_Manager::HSC_Manager(){

}


HSC_Manager::~HSC_Manager(){

}

/**
 * Initialize the manager and set main parameters
 */
hscError HSC_Manager::Setup(){
	_solver.Setup();

	return  NO_ERROR;
}


hscError HSC_Manager::InsertModel(uint key, vector<HSCModel_Base> model){
	_models[key] = model;
	return  NO_ERROR;
}


void HSC_Manager::startDeviceThreads(){
	//TODO: using scheduler to asynchronous execution
}


hscError HSC_Manager::Reinit(){
	return  NO_ERROR;
}


hscError HSC_Manager::PrepareSolver(){
	hscError res = _solver.PrepareSolver(_models);
	if( res == NO_ERROR)
		_models.clear();

	startDeviceThreads();
	return res;
}


/**
 * Create and add a task to scheduler
 */
hscError HSC_Manager::AddInputTask(hscID_t id){
	//make taskID
	HSC_TaskInfo task;
	task.modelPack = _solver.LocateDataByID(id);

	//Add to scheduler
	_scheduler.AddInputTask(task);
	return  NO_ERROR;
}

/**
 * A time increment process for each object
 */
hscError HSC_Manager::Process(HSC_TaskInfo * task, HSC_Device* d){
	task = _scheduler.GetInputTask();
	_solver.Process(task->modelPack, d);
	return  NO_ERROR;
}


//#ifdef DO_UNIT_TESTS
#include <cassert>

using namespace std;

void testHSC_manager()
{
	cout << "testHSC_manager" << endl <<flush;

	HSC_Manager manager;

	//Setup Manager
	assert(manager.Setup() == NO_ERROR);

	//Create HHChannels and check execution
	int modelNumber = 100;
	int modelSize = 100;

	uint id = 0;
	for (int var = 0; var < modelNumber; ++var) {
		vector<HSCModel_Base> model;
		for (int channels = 0; channels < modelSize; ++channels) {
			HSCModel_HHChannel ch;
			ch.Vm = var+channels;
			ch.id = id++;

			model.push_back(ch);
		}
		manager.InsertModel(var+200,model);
	}
	manager.Reinit();
	manager.PrepareSolver();
	manager.AddInputTask(0);

	int ndev= HSC_Device::GetNumberOfActiveDevices();
	if(ndev > 0)
	{
		HSC_Device* d = new HSC_Device(1);
		manager.Process(NULL, d);
		delete d;
	}


	//Create Models
//	vector< int* > childArray;
//	vector< unsigned int > childArraySize;
//
//	/**
//	 *  Cell 4:
//	 *
//	 *             3  <--- Soma
//	 *             |
//	 *             2
//	 *            / \
//	 *           /   \
//	 *          1     0
//	 *
//	 */
//
//	int childArray_4[ ] =
//	{
//		/* c0  */  -1,
//		/* c1  */  -1,
//		/* c2  */  -1, 0, 1,
//		/* c3  */  -1, 2,
//	};
//
//	childArray.push_back( childArray_4 );
//	childArraySize.push_back( sizeof( childArray_4 ) / sizeof( int ) );
//
//	double epsilon = 1e-17;
//	unsigned int i;
//	unsigned int j;
//	unsigned int nCompt;
//	int* array;
//	unsigned int arraySize;
//
//	int cellNumber = 100;
//	for ( unsigned int cell = 0; cell < childArray.size(); cell++ ) {
//		array = childArray[ cell ];
//		arraySize = childArraySize[ cell ];
//		nCompt = count( array, array + arraySize, -1 );
//
//
//
//		assert(m.InsertModel() == NO_ERROR);
//	}
}
//#endif

