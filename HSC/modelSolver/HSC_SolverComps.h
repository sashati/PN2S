///////////////////////////////////////////////////////////
//  HSC_SolverComps.h
//  Implementation of the Class HSC_SolverComps
//  Created on:      27-Dec-2013 7:57:50 PM
//  Original author: Saeed Shariati
///////////////////////////////////////////////////////////

#if !defined(EA_ABB95B66_E531_4681_AE2B_D1CE4B940FF6__INCLUDED_)
#define EA_ABB95B66_E531_4681_AE2B_D1CE4B940FF6__INCLUDED_

#include "../Definitions.h"
#include "HSCModel.h"
#include "HSC_NetworkAnalyzer.h"

class HSC_SolverComps
{
private:
	double dt;

	//CUDA variables
	float ** _hmListOfDevice; //Contains pointers to all LU matrices in GPU
	int * _pivotArray_h; //Contains Pivots of all models in LU format at GPU

	//Just for test
	float ** _hmListOfHost; //Pointers to the Hines matrix or LU matrix at host

	void  makeHinesMatrix(HSCModel *model, float * matrix);// float** matrix, uint nCompt);

public:
	HSC_SolverComps(double _dt);
	~HSC_SolverComps();
	hscError PrepareSolver(vector< HSCModel> &models, HSC_NetworkAnalyzer &analyzer);
	hscError Process();
};
#endif // !defined(EA_ABB95B66_E531_4681_AE2B_D1CE4B940FF6__INCLUDED_)
