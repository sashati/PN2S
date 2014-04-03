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
	T *_hm; // A pointer to Hines Matrices
	T *_hm_dev; // A pointer to Hines Matrices
	T *_rhs; // Right hand side of the equation
	T *_rhs_dev; // Right hand side of the equation
	T *_Vm; // Vm of the compartments
	T *_Vm_dev; // Vm of the compartments


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
