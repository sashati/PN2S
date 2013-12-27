///////////////////////////////////////////////////////////
//  HSCModel_Compartment.h
//  Implementation of the Class HSCModel_Compartment
//  Created on:      26-Dec-2013 4:20:51 PM
//  Original author: saeed
///////////////////////////////////////////////////////////

#if !defined(EA_2FDBC007_C354_46d1_A9D9_572AE399143A__INCLUDED_)
#define EA_2FDBC007_C354_46d1_A9D9_572AE399143A__INCLUDED_

#include "HSC_CaChannel.h"
#include "HSCModel_CustomChannel.h"
#include "HSCModel_HHChannel.h"
#include "HSCModel_SynapticChannel.h"

class HSCModel_Compartment
{

public:
	HSCModel_Compartment();
	virtual ~HSCModel_Compartment();
	HSCModel_CaChannel *m_HSCModel_CaChannel;
	HSCModel_Compartment *m_HSCModel_Compartment;
	HSCModel_CustomChannel *m_HSCModel_CustomChannel;
	HSCModel_HHChannel *m_HSCModel_HHChannel;
	HSCModel_SynapticChannel *m_HSCModel_SynapticChannel;

private:
	HSCModel_Compartment back;
	HSCModel_Compartment forward;

};
#endif // !defined(EA_2FDBC007_C354_46d1_A9D9_572AE399143A__INCLUDED_)
