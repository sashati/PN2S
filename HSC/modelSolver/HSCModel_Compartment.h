///////////////////////////////////////////////////////////
//  HSCModel_Compartment.h
//  Implementation of the Class HSCModel_Compartment
//  Created on:      27-Dec-2013 9:35:18 PM
//  Original author: saeed
///////////////////////////////////////////////////////////

#if !defined(EA_F11DAA80_6F7A_4ad9_B555_12F0C681E799__INCLUDED_)
#define EA_F11DAA80_6F7A_4ad9_B555_12F0C681E799__INCLUDED_

#include "../Definitions.h"
#include "HSCModel_CaChannel.h"
#include "HSCModel_CustomChannel.h"
#include "HSCModel_HHChannel.h"
#include "HSCModel_SynapticChannel.h"

class HSCModel_Compartment
{
public:
	vector< unsigned int > children;	///< Hines indices of child compts
	double Ra;
	double Rm;
	double Cm;
	double Em;
	double initVm;

	vector<HSCModel_HHChannel> hhchannels;

	HSCModel_Compartment();
	HSCModel_Compartment(uint _id);
	virtual ~HSCModel_Compartment();

private:

};
#endif // !defined(EA_F11DAA80_6F7A_4ad9_B555_12F0C681E799__INCLUDED_)
