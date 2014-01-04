///////////////////////////////////////////////////////////
//  HSC_NetworkAnalyzer.h
//  Implementation of the Class HSC_NetworkAnalyzer
//  Created on:      30-Dec-2013 4:04:20 PM
//  Original author: Saeed Shariati
///////////////////////////////////////////////////////////

#if !defined(AA2911C45_CDD0_4e09_A1A2_A5363E6EF36B__INCLUDED_)
#define AA2911C45_CDD0_4e09_A1A2_A5363E6EF36B__INCLUDED_

#include "../Definitions.h"
#include "HSCModel.h"

class HSC_NetworkAnalyzer
{

public:
	uint nComp;
	uint nModel;
	HSC_NetworkAnalyzer();
	virtual ~HSC_NetworkAnalyzer();
	hscError ImportNetwork(vector<HSCModel> &network);

	vector<HSCModel_Compartment*> allCompartments;
	vector<HSCModel_HHChannel*> allHHChannels;
private:
	hscError importCompts(vector<HSCModel_Compartment> &cmpts);
	hscError importHHChannels(vector<HSCModel_HHChannel> &chs);

};
#endif // !defined(AA2911C45_CDD0_4e09_A1A2_A5363E6EF36B__INCLUDED_)
