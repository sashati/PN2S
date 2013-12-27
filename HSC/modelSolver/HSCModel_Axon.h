///////////////////////////////////////////////////////////
//  HSCModel_Axon.h
//  Implementation of the Class HSCModel_Axon
//  Created on:      26-Dec-2013 4:20:50 PM
//  Original author: saeed
///////////////////////////////////////////////////////////

#if !defined(EA_0D7CEB44_7D8D_4cb4_92A0_799715709F94__INCLUDED_)
#define EA_0D7CEB44_7D8D_4cb4_92A0_799715709F94__INCLUDED_

#include "HSCModel_SynapticChannel.h"

class HSCModel_Axon
{

public:
	HSCModel_Axon();
	virtual ~HSCModel_Axon();
	HSCModel_SynapticChannel *m_HSCModel_SynapticChannel;

};
#endif // !defined(EA_0D7CEB44_7D8D_4cb4_92A0_799715709F94__INCLUDED_)
