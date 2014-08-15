///////////////////////////////////////////////////////////
//  PN2S_Proxy.cpp
//  Implementation of the Class PN2S_Proxy
//  Created on:      26-Dec-2013 4:08:07 PM
//  Original author: Saeed Shariati
///////////////////////////////////////////////////////////

#include "PN2S_Proxy.h"
#include "PN2S/headers.h"
#include "PN2S/Manager.h"
#include "PN2S/core/models/SolverComps.h"
#include "HSolveUtils.h"

//For get info from Shell
#include "../headers.h"
#include "../biophysics/Compartment.h"
#include "ZombieCompartment.h"
#include "../biophysics/CaConc.h"
#include "ZombieCaConc.h"
#include "../biophysics/HHGate.h"
#include "../biophysics/ChanBase.h"
#include "../biophysics/HHChannel.h"
#include "ZombieHHChannel.h"
#include "../shell/Wildcard.h"
#include "../shell/Shell.h"
#include "HSolve.h"
#include <pthread.h>

using namespace pn2s;

extern std::map< unsigned int, pn2s::Location > locationMap; //Locates in DeviceManager


void PN2S_Proxy::FillData(map<unsigned int, Id> modelId_map){
	for (uint dev_i = 0; dev_i < DeviceManager::Devices().size(); ++dev_i) {
		pn2s::Device& dev = DeviceManager::Devices()[dev_i];
		for (uint mp_i = 0; mp_i < dev.ModelPacks().size(); ++mp_i) {
			pn2s::ModelPack& mp = dev.ModelPacks()[mp_i];
			int cmpt_idx = 0;
			int ch_idx = 0;
			for (uint m_i = 0; m_i < mp.models.size(); ++m_i) {
				Id model = modelId_map[mp.models[m_i]];

				//Add index of HSolve object
				locationMap[model.value()] = Location(dev_i,mp_i);

				//Put model into solvers
				HSolve* h =	reinterpret_cast< HSolve* >( model.eref().data());

				for(unsigned int  i = 0; i<h->nCompt_;i++)
					for(unsigned int  j = 0; j<h->nCompt_;j++)
						mp.ComptSolver().SetHinesMatrix(m_i,i,j, h->getA(i,j));

				//Compartments and channels
				vector< CurrentStruct >::iterator icurrent = h->HSolveActive::current_.begin();
				typedef vector< CurrentStruct >::iterator currentVecIter;
				vector< currentVecIter >::iterator iboundary = h->HSolveActive::currentBoundary_.begin();

				vector< double >::iterator istate = h->HSolveActive::state_.begin();
				vector< ChannelStruct >::iterator ichannel = h->HSolveActive::channel_.begin();


				for ( uint ic = 0; ic < h->HSolvePassive::compartmentId_.size(); ++ic, iboundary++, cmpt_idx++ )
				{
					Id cc = h->HSolvePassive::compartmentId_[ ic ];
					//Add to location map
					locationMap[cc.value()] = Location(dev_i,mp_i,cmpt_idx);

					//Copy Data
					mp.ComptSolver().SetValue(cmpt_idx,FIELD::VM,h->getVm(cc));
					mp.ComptSolver().SetValue(cmpt_idx,FIELD::CONSTANT,h->getHS2(cc));
					mp.ComptSolver().SetValue(cmpt_idx,FIELD::INIT_VM,h->getInitVm(cc));
					mp.ComptSolver().SetValue(cmpt_idx,FIELD::RA,h->getRa(cc));
					mp.ComptSolver().SetValue(cmpt_idx,FIELD::CM_BY_DT,h->getCmByDt(cc));
					mp.ComptSolver().SetValue(cmpt_idx,FIELD::EM_BY_RM,h->getEmByRm(cc));

					mp.ComptSolver().SetValue(cmpt_idx,FIELD::INJECT_BASAL,h->getInject(cc));

					//Add Channel for a compartment
					for ( ; icurrent < *iboundary; ++icurrent, ch_idx++, ichannel++ )
					{
						if(ichannel->Xpower_)
						{
							mp.ChannelSolver().SetValue(ch_idx,FIELD::CH_X_POWER,ichannel->Xpower_);
							mp.ChannelSolver().SetValue(ch_idx,FIELD::CH_X,*istate);
							mp.ChannelSolver().SetGateXParams(ch_idx,ichannel->Xparams);
							istate++;
						}
						if(ichannel->Ypower_)
						{
							mp.ChannelSolver().SetValue(ch_idx,FIELD::CH_Y_POWER,ichannel->Ypower_);
							mp.ChannelSolver().SetValue(ch_idx,FIELD::CH_Y,*istate);
							mp.ChannelSolver().SetGateYParams(ch_idx,ichannel->Yparams);
							istate++;
						}
						if(ichannel->Zpower_)
						{
							mp.ChannelSolver().SetValue(ch_idx,FIELD::CH_Z_POWER,ichannel->Zpower_);
							mp.ChannelSolver().SetValue(ch_idx,FIELD::CH_Z,*istate);
							mp.ChannelSolver().SetGateZParams(ch_idx,ichannel->Zparams);
							istate++;
						}
						mp.ChannelSolver().SetValue(ch_idx,FIELD::CH_GBAR,ichannel->Gbar_);
						mp.ChannelSolver().SetValue(ch_idx,FIELD::CH_GK,icurrent->Gk);
						mp.ChannelSolver().SetValue(ch_idx,FIELD::CH_EK,icurrent->Ek);

						//Connect the channel with the compartment
						mp.ComptSolver().ConnectChannel(cmpt_idx,ch_idx);
						mp.ChannelSolver().SetValue(ch_idx,FIELD::CH_COMPONENT_INDEX,cmpt_idx);

					}
				}
			}
		}
	}
}

void PN2S_Proxy::Reinit(Eref hsolve){
	//Create model structures and Allocate memory
	Manager::Allocate();

	//create a map from Id to Location
//	typename vector<Device>::iterator dev;
//	typename vector<ModelPack>::iterator mp;
//	for( dev = DeviceManager::_device.begin(); dev != DeviceManager::_device.end(); ++dev) {
//		for( mp = dev->_modelPacks.begin(); mp != dev->_modelPacks.end(); ++mp) {
//			for (size_t m = 0; m < mp->stat.nModels; ++m) {
//				models::Model& model = mp->models[m];
//				for (size_t c = 0; c < mp->stat.nCompts; ++c) {
//					models::Compartment* cmp = &(model.compts[c]);
//					_compartmentMap[cmp->gid] = cmp->location;
//				}
//			}
//		}
//	}

	/**
	 * Zumbify and Copy data values
	 */
    vector< Id >::const_iterator i;
	vector< ObjId > temp;

//    for ( i = _all_compartmentIds.begin(); i != _all_compartmentIds.end(); ++i )
//		temp.push_back( ObjId( *i, 0 ) );
	Shell::dropClockMsgs( temp, "init" );
	Shell::dropClockMsgs( temp, "process" );
//    for ( i = _all_compartmentIds.begin(); i != _all_compartmentIds.end(); ++i )
//        CompartmentBase::zombify( i->eref().element(),
//					   ZombieCompartment::initCinfo(), hsolve.id() );
    //	temp.clear();
    //    for ( i = caConcId_.begin(); i != caConcId_.end(); ++i )
    //		temp.push_back( ObjId( *i, 0 ) );
    //	Shell::dropClockMsgs( temp, "process" );
    //    for ( i = caConcId_.begin(); i != caConcId_.end(); ++i )
    //        ZombieCaConc::zombify( hsolve.element(), i->eref().element() );
    //
    //	temp.clear();
    //    for ( i = channelId_.begin(); i != channelId_.end(); ++i )
    //		temp.push_back( ObjId( *i, 0 ) );
    //	Shell::dropClockMsgs( temp, "process" );
    //    for ( i = channelId_.begin(); i != channelId_.end(); ++i )
    //        ZombieHHChannel::zombify( hsolve.element(), i->eref().element() );

    //Prepare solvers
	Manager::PrepareSolvers();
}

/**
 * Interface Set/Get functions
 */
extern std::map< unsigned int, pn2s::Location > locationMap;


void PN2S_Proxy::AddExternalCurrent( unsigned int id, TYPE_ Gk, TYPE_ GkEk)
{
	Location l = locationMap[id];
	DeviceManager::Devices()[l.device].ModelPacks()[l.pack]._compsSolver.AddExternalCurrent(l.index,Gk, GkEk);
}

void PN2S_Proxy::setValue( unsigned int id, TYPE_ value , FIELD::TYPE n)
{
	Location l = locationMap[id];
	DeviceManager::Devices()[l.device].ModelPacks()[l.pack]._compsSolver.SetValue(l.index,n,value);
}

TYPE_ PN2S_Proxy::getValue( unsigned int id, FIELD::TYPE n)
{
	Location l = locationMap[id];
	return DeviceManager::Devices()[l.device].ModelPacks()[l.pack]._compsSolver.GetValue(l.index,n);
}
