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

struct HSC_HinexMatrix
{

};

class HSC_SolverComps
{
private:
	double dt;

public:
	HSC_SolverComps(double _dt);
	hscError PrepareSolver(vector< HSCModel> &models, HSC_NetworkAnalyzer &analyzer);
	void  makeHinesMatrix(HSCModel *model, HSC_HinexMatrix& matrix);
};
#endif // !defined(EA_ABB95B66_E531_4681_AE2B_D1CE4B940FF6__INCLUDED_)
