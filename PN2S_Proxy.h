///////////////////////////////////////////////////////////
//  PN2S_Proxy.h
//  Implementation of the Class PN2S_Proxy
//  Created on:      26-Dec-2013 4:08:07 PM
//  Original author: Saeed Shariati
///////////////////////////////////////////////////////////

#pragma once

/**
 * The class that use PN2S package and deal with moose constrains.
 */
#include "PN2S/headers.h"
#include "header.h" //Moose parts
#include "HinesMatrix.h"


class PN2S_Proxy
{
private:
	static void fillData(map<unsigned int, Id> &modelId_map);
	static void readSynapses(vector< Id >	&compartmentId_);
//	static void manageOutgoingMessages(vector< Id >	&compartmentId_);
public:
	static void Process(ProcPtr info);
	static void Reinit(map<unsigned int, Id> modelId_map);
	static void ModelDistribution(pn2s::Model_pack_info& m, double dt);

	static void setValue( unsigned int , TYPE_, pn2s::FIELD::CM);
	static TYPE_ getValue( unsigned int , pn2s::FIELD::CM);
	static void setValue( unsigned int , TYPE_, pn2s::FIELD::GATE);
	static TYPE_ getValue( unsigned int , pn2s::FIELD::GATE);
//	static void setValue( unsigned int , TYPE_, pn2s::FIELD::CH);
//	static TYPE_ getValue( unsigned int , pn2s::FIELD::CH);

	static void Initialize();
	static void Close();

};
