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

template <typename T, int arch>
class HSC_SolverComps
{
private:
	double _dt;
	uint nModel;
	uint nComp;


	//CUDA variables
	T **_hmList_dev; // A pointer to the Device list of matrix pointers
	T ** _hmListOfMatricesInDevice; //A list in Host that each node is a device pointers to one LU matrix
	int * _pivotArray_h; //Contains Pivots of all models in LU format at GPU
	int * _pivotArray_d;
	int * _infoArray_d;
	int * _infoArray_h;

	//Just for test
	T ** _hmListOfHost; //Pointers to the Hines matrix or LU matrix at host

	void  makeHinesMatrix(HSCModel *model, T * matrix);// float** matrix, uint nCompt);

public:
	HSC_SolverComps();
	~HSC_SolverComps();
	hscError PrepareSolver(vector< HSCModel> &models, HSC_NetworkAnalyzer &analyzer);
	hscError Process();

	double GetDt(){ return _dt;}
	void SetDt(double dt){ _dt = dt;}

};
#endif // !defined(EA_ABB95B66_E531_4681_AE2B_D1CE4B940FF6__INCLUDED_)
