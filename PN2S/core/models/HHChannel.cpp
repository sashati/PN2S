///////////////////////////////////////////////////////////
//  HHChannel.cpp
//  Implementation of the Class HHChannel
//  Created on:      26-Dec-2013 4:20:54 PM
//  Original author: Saeed Shariati
///////////////////////////////////////////////////////////

#include "HHChannel.h"
#include "../../headers.h"

using namespace pn2s::models;

template <typename T, int arch>
HHChannel<T,arch>::HHChannel(){
}

template <typename T, int arch>
HHChannel<T,arch>::~HHChannel(){
}

template class HHChannel<double, ARCH_SM30>;
