#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <math.h>
#include <iterator>
#include <vector>
#include <ctime>
#include <math_functions.h>


#include <iostream>
#include <cstdlib>

#include "../PN2S/core/models/SolverComps.h"
#include "../PN2S/core/models/SolverChannels.h"
#include "../PN2S/headers.h"
#include "../PN2S/Manager.h"

//#include "../../headers.h"


using namespace pn2s;
using namespace pn2s::models;

int main(int argc, char *argv[] )
{
	double A[3][3] = {
			{1.3064676977022927e-06,	-4.97678044133036e-08,	0},
			{-4.97678044133036e-08,	2.17606820220756e-07,	-4.2169026478553e-08},
			{0,	-4.2169026478553e-08,	0.00785051835950267},
	};
	double p_x[] = {-4500.0000000000009,
						  -100000,
						  -1,
						  0.045000000000000005,
						  -0.01,
						  4000,
						  0,
						  0,
						  0.070000000000000007,
						  0.017999999999999999,
						  3001,
						  -0.10000000000000001,
						  0.049999999999999989};

	double p_y[] = {	70,
						0,
						0,
						0.070000000000000007,
						0.02,
						1000,
						0,
						1,
						0.040000000000000008,
						-0.01,
						3001,
						-0.10000000000000001,
						0.049999999999999989 };

	pn2s::DeviceManager::Initialize();

	int channelNumber = 10;
	double dt_ = 9.9999999999999995e-07;
	//Create model

	pn2s::models::ModelStatistic st;
	st.nModels = 9;
	if (argc > 1)
		st.nModels = atoi(argv[1]);
	st.nCompts_per_model = 3;
	st.nChannels_all = st.nCompts_per_model * st.nModels* channelNumber;


	vector<int2> modelST;
	vector<unsigned int> modelIds;

	for (int i = 0; i < st.nModels; ++i) {
		int2 st;
		st.x = 3;
		st.y = st.x * channelNumber;
		modelST.push_back(st);
		modelIds.push_back(i);
	}
	pn2s::DeviceManager::AllocateMemory(modelIds,modelST,dt_);

	//Fill data
	for (uint dev_i = 0; dev_i < DeviceManager::Devices().size(); ++dev_i) {
		pn2s::Device& dev = DeviceManager::Devices()[dev_i];
		for (uint mp_i = 0; mp_i < dev.ModelPacks().size(); ++mp_i) {
			pn2s::ModelPack& mp = dev.ModelPacks()[mp_i];
			int cmpt_idx = 0;
			int ch_idx = 0;
			for (uint m_i = 0; m_i < mp.models.size(); ++m_i) {

				for(int compt = 0; compt<st.nCompts_per_model;compt++, cmpt_idx++)
				{
					for(int j = 0; j<st.nCompts_per_model;j++)
						mp.ComptSolver().SetHinesMatrix(m_i,compt,j, A[compt][j]);

					mp.ComptSolver().SetValue(cmpt_idx,FIELD::VM,-0.070000000000000007);
					mp.ComptSolver().SetValue(cmpt_idx,FIELD::CONSTANT,1.3064676977022927e-06);
					mp.ComptSolver().SetValue(cmpt_idx,FIELD::INIT_VM,-0.070000000000000007);
					mp.ComptSolver().SetValue(cmpt_idx,FIELD::RA,397887.35772973846);
					mp.ComptSolver().SetValue(cmpt_idx,FIELD::CM_BY_DT,1.2566370614359173e-06);
					mp.ComptSolver().SetValue(cmpt_idx,FIELD::EM_BY_RM,-4.3982297150257105e-12);
					mp.ComptSolver().SetValue(cmpt_idx,FIELD::INJECT_BASAL,0);

					if (compt % 3 == 0)
						channelNumber = 0;
					else if (compt % 3 == 1)
						channelNumber = 2;
					else
						channelNumber = 8;

					for (int i = 0; i < channelNumber; ++i, ch_idx++) {
						mp.ChannelSolver().SetValue(i,FIELD::CH_X_POWER,3);
						mp.ChannelSolver().SetValue(i,FIELD::CH_X,0.052932485257228955);
						mp.ChannelSolver().SetGateXParams(i,vector<double> (p_x, p_x + sizeof p_x / sizeof p_x[0] ));

						mp.ChannelSolver().SetValue(i,FIELD::CH_Y_POWER,1);
						mp.ChannelSolver().SetValue(i,FIELD::CH_Y,0.59612075350857563);
						mp.ChannelSolver().SetGateYParams(i,vector<double> (p_y, p_y + sizeof p_y / sizeof p_y[0] ));

						mp.ChannelSolver().SetValue(i,FIELD::CH_GBAR,0.00094200000000000002);
						mp.ChannelSolver().SetValue(i,FIELD::CH_GK,0);
						mp.ChannelSolver().SetValue(i,FIELD::CH_EK,0.044999999999999998);

						mp.ComptSolver().ConnectChannel(cmpt_idx,ch_idx);
						mp.ChannelSolver().SetValue(ch_idx,FIELD::CH_COMPONENT_INDEX,cmpt_idx);
					}
				}
				mp.ComptSolver().SetValue(cmpt_idx-1,FIELD::INJECT_BASAL,9.9999999999999995e-08);
			}
		}
	}

	pn2s::DeviceManager::PrepareSolvers();



//	int ncomp = st.nCompts_per_model* st.nModels;

//	mp.ComptSolver()._threads=dim3(128);
//	mp.ComptSolver()._blocks=dim3(ncomp / mp.ComptSolver()._threads.x + 1);
//
//
	timespec ts_beg, ts_end;
	double tt = 0;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts_beg);
	for (int i = 0; i < 100; ++i) {
		pn2s::DeviceManager::Process();
	}
	cudaDeviceSynchronize();
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts_end);
	tt += (ts_end.tv_sec - ts_beg.tv_sec) + (ts_end.tv_nsec - ts_beg.tv_nsec) / 1e9;
	cout
		<< std::setprecision (std::numeric_limits< double >::digits10) <<tt/100.0 * 1000000 << " msec" << "\t "
//		<< "Thread:(" << mp.ComptSolver()._threads.x << "," << mp.ComptSolver()._threads.y << "," << mp.ComptSolver()._threads.z <<")\t "
//			"Block:(" << mp.ComptSolver()._blocks.x << "," << mp.ComptSolver()._blocks.y << "," << mp.ComptSolver()._blocks.z <<")"
//			<< "\t nComp: " << ncomp
			<< endl <<flush;
	return 0;
}

