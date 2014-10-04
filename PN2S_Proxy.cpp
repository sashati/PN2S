///////////////////////////////////////////////////////////
//  PN2S_Proxy.cpp
//  Implementation of the Class PN2S_Proxy
//  Created on:      26-Dec-2013 4:08:07 PM
//  Original author: Saeed Shariati
///////////////////////////////////////////////////////////

#include "PN2S_Proxy.h"
#include "PN2S/headers.h"
#include "PN2S/ResourceManager.h"
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

std::map< pn2s::Location, Eref > spikegen_;

void PN2S_Proxy::Process(ProcPtr info){
	ResourceManager::Process();

	//Send Spikes
	typedef std::map<pn2s::Location, Eref>::iterator it_type;
	for(it_type it = spikegen_.begin(); it != spikegen_.end(); it++) {
		pn2s::Location l = it->first;
		double vm =	ResourceManager::GetValue(l,pn2s::FIELD::VM);
		SpikeGen* spike = reinterpret_cast< SpikeGen* >( it->second.data() );
		spike->handleVm( vm );
		spike->process( it->second, info );
	}
}

void PN2S_Proxy::fillData(map<unsigned int, Id> &modelId_map){
	vector<Device*> devices = ResourceManager::Devices();
	for (uint dev_i = 0; dev_i < devices.size(); ++dev_i) {
		pn2s::Device* dev = devices[dev_i];
		for (uint mp_i = 0; mp_i < dev->ModelPacks().size(); ++mp_i) {
			pn2s::ModelPack& mp = dev->ModelPacks()[mp_i];
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

					int gate_local_index = 0;
					//Add Channel for a compartment
					for ( ; icurrent < *iboundary; ++icurrent, ch_idx++, ichannel++, gate_local_index++ )
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
				readSynapses(h->HSolvePassive::compartmentId_);
			}
		}
	}
}

void PN2S_Proxy::Reinit(map<unsigned int, Id> modelId_map){
	spikegen_.clear();
	fillData(modelId_map);
	ResourceManager::PrepareSolvers();
}

void PN2S_Proxy::ModelDistribution(pn2s::Model_pack_info& m, double dt)
{
	ResourceManager::ModelDistribution(m,dt);
}

void PN2S_Proxy::readSynapses(vector< Id >	&compartmentId_)
{
    vector< Id > spikeId;
    vector< Id > synId;
    vector< Id >::iterator syn;
    vector< Id >::iterator spike;
    SynChanStruct synchan;

    for ( unsigned int ic = 0; ic < compartmentId_.size(); ++ic )
    {
        synId.clear();
        HSolveUtils::synchans( compartmentId_[ ic ], synId );
        for ( syn = synId.begin(); syn != synId.end(); ++syn )
        {
            synchan.compt_ = ic;
            synchan.elm_ = *syn;
        }

        static const Finfo* procDest = SpikeGen::initCinfo()->findFinfo( "process");
        assert( procDest );
        const DestFinfo* df = dynamic_cast< const DestFinfo* >( procDest );
        assert( df );

        spikeId.clear();
        HSolveUtils::spikegens( compartmentId_[ ic ], spikeId );
        // Very unlikely that there will be >1 spikegens in a compartment,
        // but lets take care of it anyway.
        for ( spike = spikeId.begin(); spike != spikeId.end(); ++spike )
        {
        	Location l = locationMap[compartmentId_[ ic ].value() ];
            spikegen_[l] = spike->eref();

            ObjId mid = spike->element()->findCaller( df->getFid() );
            if ( ! mid.bad()  )
                Msg::deleteMsg( mid );
        }
    }
}

void PN2S_Proxy::Initialize()
{
	ResourceManager::Initialize();
}

void PN2S_Proxy::Close()
{
	ResourceManager::Close();
}

/**
 * Interface Set/Get functions
 */
extern std::map< unsigned int, pn2s::Location > locationMap;

void PN2S_Proxy::setValue( unsigned int id, TYPE_ value , FIELD::CM n)
{
	ResourceManager::SetValue(locationMap[id], n,value);
}

TYPE_ PN2S_Proxy::getValue( unsigned int id, FIELD::CM n)
{
	return 	ResourceManager::GetValue(locationMap[id], n);
}

void PN2S_Proxy::setValue( unsigned int id, TYPE_ value , FIELD::CH n)
{
	ResourceManager::SetValue(locationMap[id], n,value);
}

TYPE_ PN2S_Proxy::getValue( unsigned int id, FIELD::CH n)
{
	return 	ResourceManager::GetValue(locationMap[id], n);
}
