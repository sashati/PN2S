///////////////////////////////////////////////////////////
//  HSC_SolverChannels.cpp
//  Implementation of the Class HSC_SolverChannels
//  Created on:      27-Dec-2013 4:23:16 PM
//  Original author: Saeed Shariati
///////////////////////////////////////////////////////////

#include "HSC_SolverChannels.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

HSC_SolverChannels::HSC_SolverChannels(){

}

HSC_SolverChannels::~HSC_SolverChannels(){
	free(hostMemory);
}

static __inline__ void modify (cublasHandle_t handle, float *m, int ldm, int n, int p, int q, float alpha, float beta){
    cublasSscal (handle, n-p, &alpha, &m[IDX2C(p,q,ldm)], ldm);
    cublasSscal (handle, ldm-p, &beta, &m[IDX2C(p,q,ldm)], 1);
}


hscError HSC_SolverChannels::PrepareSolver(map<hscID_t, vector <HSCModel_Base> > &models,	HSCModelStatistic st){
	uint hhSize = st.GetSumHHChannels();
	hostMemory = (float *)malloc (hhSize * sizeof (*hostMemory));




//	free(hostMemory)

//	uint modelNumeber = models.size();
//	for (int i = 0; i< modelNumeber; i++) {
//		uint modelSize = models[i].size();
//		for (int j = 0; j < modelSize ; j++) {
//			switch(models[i][j].type)
//			{
//			case HSCModel_Base::Type::HHCHannel:
//
//				break;
//			}
//		}
//	}

	return  NO_ERROR;
}
