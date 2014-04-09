///////////////////////////////////////////////////////////
//  HSCModel_Compartment.cpp
//  Implementation of the Class HSCModel_Compartment
//  Created on:      27-Dec-2013 9:35:18 PM
//  Original author: saeed
///////////////////////////////////////////////////////////

#include "HSCModel_Compartment.h"
//#include "../Definitions.h"

HSCModel_Compartment::HSCModel_Compartment(){
}


HSCModel_Compartment::~HSCModel_Compartment(){

}

//Copy constractor
HSCModel_Compartment::HSCModel_Compartment( const HSCModel_Compartment& other ) :
		Ra(other.Ra), Rm(other.Rm), Cm(other.Cm), Em(other.Em), initVm(other.initVm), Vm(other.Vm)
{
	children.assign(other.children.begin(), other.children.end());
}
