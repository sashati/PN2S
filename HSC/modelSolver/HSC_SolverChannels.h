///////////////////////////////////////////////////////////
//  HSC_SolverChannels.h
//  Implementation of the Class HSC_SolverChannels
//  Created on:      27-Dec-2013 4:23:16 PM
//  Original author: Saeed Shariati
///////////////////////////////////////////////////////////

#if !defined(A8661C97F_679E_4bb9_84D8_5EEB3718169D__INCLUDED_)
#define A8661C97F_679E_4bb9_84D8_5EEB3718169D__INCLUDED_
#include "../Definitions.h"
#include "HSCModel_Base.h"

class HSC_SolverChannels
{
private:

public:
	HSC_SolverChannels();
	virtual ~HSC_SolverChannels();

	hscError PrepareSolver(vector<vector<HSCModel_Base> > &models);
};
#endif // !defined(A8661C97F_679E_4bb9_84D8_5EEB3718169D__INCLUDED_)
