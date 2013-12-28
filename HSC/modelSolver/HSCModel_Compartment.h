///////////////////////////////////////////////////////////
//  HSCModel_Compartment.h
//  Implementation of the Class HSCModel_Compartment
//  Created on:      27-Dec-2013 9:35:18 PM
//  Original author: saeed
///////////////////////////////////////////////////////////

#if !defined(EA_F11DAA80_6F7A_4ad9_B555_12F0C681E799__INCLUDED_)
#define EA_F11DAA80_6F7A_4ad9_B555_12F0C681E799__INCLUDED_

#include "HSCModel_CaChannel.h"
#include "HSCModel_CustomChannel.h"
#include "HSCModel_HHChannel.h"
#include "HSCModel_SynapticChannel.h"

class HSCModel_Compartment
{

public:
	HSCModel_CaChannel* m_HSCModel_CaChannel;
	HSCModel_Compartment* m_HSCModel_Compartment;
	HSCModel_CustomChannel* m_HSCModel_CustomChannel;
	HSCModel_HHChannel* m_HSCModel_HHChannel;
	HSCModel_SynapticChannel* m_HSCModel_SynapticChannel;

	HSCModel_Compartment();
	virtual ~HSCModel_Compartment();

private:
	HSCModel_Compartment* back;
	HSCModel_Compartment* forward;

};
#endif // !defined(EA_F11DAA80_6F7A_4ad9_B555_12F0C681E799__INCLUDED_)
