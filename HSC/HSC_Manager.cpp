///////////////////////////////////////////////////////////
//  HSC_Manager.cpp
//  Implementation of the Class HSC_Manager
//
//  Created on:      26-Dec-2013 4:18:17 PM
//  Original author: Saeed Shariati
///////////////////////////////////////////////////////////

#include "Definitions.h"
#include "HSC_Manager.h"
#include "modelSolver/HSCModel.h"

HSC_Manager::HSC_Manager(){

}


HSC_Manager::~HSC_Manager(){

}

/**
 * Initialize the manager and set main parameters
 */
hscError HSC_Manager::Setup(){
	HSC_Solver::Setup();

	return  NO_ERROR;
}


hscError HSC_Manager::InsertModel(HSCModel &model){
	_models.push_back(model);
	return  NO_ERROR;
}

//
//void HSC_Manager::startDeviceThreads(){
//
//}
//
//
hscError HSC_Manager::Reinit(){
	return  NO_ERROR;
}


hscError HSC_Manager::PrepareSolver(){
	hscError res = HSC_Solver::CreateModelPack(_models);
	if( res == NO_ERROR)
		_models.clear();
	return res;
}


///**
// * A time increment process for each object
// */
//hscError HSC_Manager::Process(){
//
//	return  NULL;
//}

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
	int modelSize = 100;
	for (int var = 0; var < modelSize; ++var) {
		HSCModel_HHChannel ch;
		ch.Vm = var;

		HSCModel n;
		n.AddElement(ch);

		manager.InsertModel(n);
	}
	manager.Reinit();
	manager.PrepareSolver();


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
