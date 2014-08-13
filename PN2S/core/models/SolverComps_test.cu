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

#include "SolverComps.h"
#include "SolverChannels.h"
#include "../../headers.h"

using namespace pn2s;
using namespace pn2s::models;

int main(int argc, char *argv[] )
{
	double A[3][3] = {
			{1.3064676977022927e-06,	-4.97678044133036e-08,	0},
			{-4.97678044133036e-08,	2.17606820220756e-07,	-4.2169026478553e-08},
			{0,	-4.2169026478553e-08,	0.00785051835950267},
	};

	pn2s::models::ModelStatistic st;
	st.nModels = 10;
	st.nCompts_per_model = 3;
	st.nChannels_all = 100000;

	if (argc > 1)
		st.nModels = atoi(argv[1]);
	SolverChannels ch;
	SolverComps c;
	ch.AllocateMemory(st,'\0');
	c.AllocateMemory(st,'\0');

	//Fill data
	int cmpt_idx = 0;
	for (int m = 0; m < st.nModels; ++m) {
		for(int compt = 0; compt<st.nCompts_per_model;compt++, cmpt_idx++)
		{
			for(int j = 0; j<st.nCompts_per_model;j++)
				c.SetHinesMatrix(m,compt,j, A[compt][j]);

			c.SetValue(cmpt_idx,FIELD::VM,-0.070000000000000007);
			c.SetValue(cmpt_idx,FIELD::CONSTANT,1.3064676977022927e-06);
			c.SetValue(cmpt_idx,FIELD::INIT_VM,-0.070000000000000007);
			c.SetValue(cmpt_idx,FIELD::RA,397887.35772973846);
			c.SetValue(cmpt_idx,FIELD::CM_BY_DT,1.2566370614359173e-06);
			c.SetValue(cmpt_idx,FIELD::EM_BY_RM,-4.3982297150257105e-12);
			c.SetValue(cmpt_idx,FIELD::INJECT_BASAL,0);
		}
		c.SetValue(cmpt_idx-1,FIELD::INJECT_BASAL,9.9999999999999995e-08);
	}


	c.PrepareSolver(ch.GetFieldChannelCurrents(), ch.GetFieldVm());


	int ncomp = st.nCompts_per_model* st.nModels;

	c._threads=dim3(128);
	c._blocks=dim3(ncomp / c._threads.x + 1);


	timespec ts_beg, ts_end;

	c.Input();
	c.Process();
	c.Output();

	double tt = 0;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts_beg);
	for (int i = 0; i < 1000; ++i) {
		c.Input();
		c.Process();
		c.Output();
	}
	cudaDeviceSynchronize();
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts_end);
	tt += (ts_end.tv_sec - ts_beg.tv_sec) + (ts_end.tv_nsec - ts_beg.tv_nsec) / 1e9;
	cout
		<< std::setprecision (std::numeric_limits< double >::digits10) <<tt/1000.0 * 1000000 << " msec" << "\t "
		<< "Thread:(" << c._threads.x << "," << c._threads.y << "," << c._threads.z <<")\t "
			"Block:(" << c._blocks.x << "," << c._blocks.y << "," << c._blocks.z <<")"
			<< "\t nComp: " << ncomp
			<< endl <<flush;
	return 0;
}

