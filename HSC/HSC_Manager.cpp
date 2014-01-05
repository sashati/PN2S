///////////////////////////////////////////////////////////
//  HSC_Manager.cpp
//  Implementation of the Class HSC_Manager
//
//  Created on:      26-Dec-2013 4:18:17 PM
//  Original author: Saeed Shariati
///////////////////////////////////////////////////////////

#include "HSC_Manager.h"

HSC_Manager::HSC_Manager(){

}


HSC_Manager::~HSC_Manager(){

}

/**
 * Initialize the manager and set main parameters
 */
hscError HSC_Manager::Setup(double dt){
	_dt = dt;
	_solver.Setup(dt);

	return  NO_ERROR;
}


void HSC_Manager::startDeviceThreads(){
	//TODO: using scheduler to asynchronous execution
}


hscError HSC_Manager::Reinit(){
	return  NO_ERROR;
}


hscError HSC_Manager::PrepareSolver(){
	_deviceManager.Setup();
	hscError res = _solver.PrepareSolver(models);
//	if( res == NO_ERROR)
//		models.clear();

//	startDeviceThreads();
	return res;
}


/**
 * Create and add a task to scheduler
 */
hscError HSC_Manager::AddInputTask(uint id){
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
hscError HSC_Manager::Process(uint id){
	HSC_TaskInfo tInfo;
	tInfo.modelPack = _solver.LocateDataByID(id);
	_scheduler.AddInputTask(tInfo);

	//TODO: This part should do Asynchronously
	HSC_TaskInfo* t = _scheduler.GetInputTask();

	_solver.Process(t->modelPack, &(_deviceManager._devices[0]));
	return  NO_ERROR;
}


//#ifdef DO_UNIT_TESTS
#include <cassert>

#include "modelSolver/HSCModel_HHChannel.h"
#include "modelSolver/HSCModel_Compartment.h"

void generateModel(uint nCompt, uint arraySize, int* array,
		HSC_Manager& manager, int id) {

	HSCModel neutral(id);
	for (uint i = 0; i < nCompt; ++i) {
		HSCModel_Compartment cmp;
		cmp.Ra = 15.0 + 3.0 * i+id;
		cmp.Rm = 45.0 + 15.0 * i+id;
		cmp.Cm = 500.0 + 200.0 * i * i+id;
		neutral.compts.push_back(cmp);
	}
	int count = -1;
	for (unsigned int a = 0; a < arraySize; a++)
		if (array[a] == -1)
			count++;
		else
			neutral.compts[count].children.push_back(array[a]);
	manager.models.push_back(neutral);
}


void testHSC_manager()
{
	cout << "testHSC_manager" << endl <<flush;

	HSC_Manager manager;

	//Setup Manager
	assert(manager.Setup(1) == NO_ERROR);


	/*********************
	 * Check Hines Solver
	 *********************/
	vector< int* > childArray;
	vector< unsigned int > childArraySize;

	/**
	 *  Cell 2:
	 *
	 *             3
	 *             |
	 *   Soma--->  2
	 *            / \
	 *           /   \
	 *          1     0
	 *
	 */

	int childArray_2[ ] =
	{
		/* c0  */  -1,
		/* c1  */  -1,
		/* c2  */  -1, 0, 1, 3,
		/* c3  */  -1,
	};

	childArray.push_back( childArray_2 );
	childArraySize.push_back( sizeof( childArray_2 ) / sizeof( int ) );

	/**
	 *  Cell 3:
	 *
	 *             3
	 *             |
	 *             2
	 *            / \
	 *           /   \
	 *          1     0  <--- Soma
	 *
	 */

	int childArray_3[ ] =
	{
		/* c0  */  -1, 2,
		/* c1  */  -1,
		/* c2  */  -1, 1, 3,
		/* c3  */  -1,
	};

	childArray.push_back( childArray_3 );
	childArraySize.push_back( sizeof( childArray_3 ) / sizeof( int ) );

	/**
	 *  Cell 4:
	 *
	 *             3  <--- Soma
	 *             |
	 *             2
	 *            / \
	 *           /   \
	 *          1     0
	 *
	 */

	int childArray_4[ ] =
	{
		/* c0  */  -1,
		/* c1  */  -1,
		/* c2  */  -1, 0, 1,
		/* c3  */  -1, 2,
	};

	childArray.push_back( childArray_4 );
	childArraySize.push_back( sizeof( childArray_4 ) / sizeof( int ) );

	unsigned int nCompt;
	int* array;
	unsigned int arraySize;
	for (unsigned int cell = 0; cell < childArray.size(); cell++) {
		array = childArray[ cell ];
		arraySize = childArraySize[ cell ];
		nCompt = count( array, array + arraySize, -1 );
		generateModel(nCompt, arraySize, array, manager,cell+10);
	}


	manager.Reinit();
	manager.PrepareSolver();
	manager.Process(10);

//	int ndev= HSC_Device::GetNumberOfActiveDevices();
//	if(ndev > 0)
//	{
//		HSC_Device* d = new HSC_Device(1);
//		manager.Process(NULL, d);
//		delete d;
//	}

}


