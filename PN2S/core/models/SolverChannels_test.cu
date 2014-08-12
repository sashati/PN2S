///////////////////////////////////////////////////////////
//  SolverChannels.cpp
//  Implementation of the Class SolverChannels
//  Created on:      27-Dec-2013 7:57:50 PM
//  Original author: Saeed Shariati
///////////////////////////////////////////////////////////


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

#include "SolverChannels.h"
#include "../../headers.h"

using namespace pn2s;
using namespace pn2s::models;

int main()
{
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

	pn2s::models::ModelStatistic st;
	st.nChannels_all = 100000;

	SolverChannels ch;
	ch.AllocateMemory(st,'\0');

	//Fill data
	for (int i = 0; i < st.nChannels_all; ++i) {
		ch.SetValue(i,FIELD::CH_X_POWER,3);
		ch.SetValue(i,FIELD::CH_X,0.052932485257228955);
		ch.SetGateXParams(i,vector<double> (p_x, p_x + sizeof p_x / sizeof p_x[0] ));

		ch.SetValue(i,FIELD::CH_Y_POWER,1);
		ch.SetValue(i,FIELD::CH_Y,0.59612075350857563);
		ch.SetGateYParams(i,vector<double> (p_y, p_y + sizeof p_y / sizeof p_y[0] ));

		ch.SetValue(i,FIELD::CH_GBAR,0.00094200000000000002);
		ch.SetValue(i,FIELD::CH_GK,0);
		ch.SetValue(i,FIELD::CH_EK,0.044999999999999998);
	}


	ch.PrepareSolver();


	int threadSize = min(max((int)st.nChannels_all/8,16), 32);//max((int)st.nChannels_all/8, 16);
	ch._threads=dim3(2,threadSize, 1);
	ch._blocks=dim3(max((int)(ceil(st.nChannels_all / ch._threads.y)),1), 1);


	timespec ts_beg, ts_end;

	ch.Input();
	ch.Process();
	ch.Output();

	double tt = 0;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts_beg);
	for (int i = 0; i < 1000; ++i) {
		ch.Input();
		ch.Process();
		ch.Output();
	}
	cudaDeviceSynchronize();
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts_end);
	tt += (ts_end.tv_sec - ts_beg.tv_sec) + (ts_end.tv_nsec - ts_beg.tv_nsec) / 1e9;
	cout
		<< std::setprecision (std::numeric_limits< double >::digits10) <<tt/1000.0 * 1000000 << " msec" << "\t "
		<< "Thread:(" << ch._threads.x << "," << ch._threads.y << "," << ch._threads.z <<")\t "
			"Block:(" << ch._blocks.x << "," << ch._blocks.y << "," << ch._blocks.z <<")"<< endl <<flush;
	return 0;
}

