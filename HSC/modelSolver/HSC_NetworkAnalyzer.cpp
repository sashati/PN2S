///////////////////////////////////////////////////////////
//  HSC_NetworkAnalyzer.cpp
//  Implementation of the Class HSC_NetworkAnalyzer
//  Created on:      30-Dec-2013 4:04:20 PM
//  Original author: Saeed Shariati
///////////////////////////////////////////////////////////

#include "HSC_NetworkAnalyzer.h"
#include <assert.h>

HSC_NetworkAnalyzer::HSC_NetworkAnalyzer(){
	allCompartments.clear();
	allHHChannels.clear();
	nComp = 0;
}


HSC_NetworkAnalyzer::~HSC_NetworkAnalyzer(){

}

hscError HSC_NetworkAnalyzer::ImportNetwork(vector<HSCModel> &network){
	hscError res;
	nModel = network.size();
	if( nModel >0)
		nComp = network[0].compts.size();

	for(vector<HSCModel>::iterator n = network.begin(); n != network.end(); ++n) {
		res = importCompts(n->compts);
		assert(res==NO_ERROR);
		res = importHHChannels(n->hhChannels);
		assert(res==NO_ERROR);
	}
	return NO_ERROR;
}

hscError HSC_NetworkAnalyzer::importCompts(vector<HSCModel_Compartment> &cmpts)
{
	for(vector<HSCModel_Compartment>::iterator n = cmpts.begin(); n != cmpts.end(); ++n) {
		allCompartments.push_back(n.base());
		importHHChannels(n->hhchannels);
	}
	return NO_ERROR;
}

hscError HSC_NetworkAnalyzer::importHHChannels(vector<HSCModel_HHChannel> &chs)
{
	if(chs.size() > 0)
	{
		for(vector<HSCModel_HHChannel>::iterator n = chs.begin(); n != chs.end(); ++n) {
			allHHChannels.push_back(n.base());
		}
	}
	return NO_ERROR;
}
