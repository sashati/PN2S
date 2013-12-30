///////////////////////////////////////////////////////////
//  HSC_SolverComps.cpp
//  Implementation of the Class HSC_SolverComps
//  Created on:      27-Dec-2013 7:57:50 PM
//  Original author: Saeed Shariati
///////////////////////////////////////////////////////////

#include "HSC_SolverComps.h"


HSC_SolverComps::HSC_SolverComps(double _dt): dt(_dt)
{
}

hscError HSC_SolverComps::PrepareSolver(vector<HSCModel > &network, HSC_NetworkAnalyzer &analyzer){

	for(vector<HSCModel >::iterator n = network.begin(); n != network.end(); n++)
	{
		HSC_HinexMatrix hm;
		makeHinesMatrix(n.base(), hm);

	}

//	uint nCompAll = analyzer.allCompartments.size();

	//	hostMemory = (float *) malloc(hhSize * sizeof(*hostMemory));
	//
	//	//Fill data
	//	float* dataPointer = hostMemory;
	//	for (uint i = 0; i < networkSize; i++) {
	//		uint modelSize = network[i].hhChannels.size();
	//		for (uint j = 0; j < modelSize; j++) {
	//			*dataPointer = network[i].hhChannels[j].Vm;
	//			dataPointer++;
	//		}
	//	}
	//
	//	cudaMalloc((void**) &deviceMemory, hhSize * sizeof(*deviceMemory));
	//
	//	cublasHandle_t cublasHandle;
	//	cublasStatus_t stat = cublasCreate(&cublasHandle);
	//
	//	cublasSetVector(hhSize, sizeof(*hostMemory), hostMemory, 1, deviceMemory,1);
	//
	//	for(float f =1.1; f<10; f+=.001)
	//		cublasSscal(cublasHandle, hhSize,&f , deviceMemory, 1);
	//
	//	cublasGetVector(hhSize, sizeof(*hostMemory), deviceMemory, 1, hostMemory,	1);
	//
	//	cudaFree(deviceMemory);
	//	cublasDestroy(cublasHandle);
	//	free(hostMemory);

	return NO_ERROR;
}



void HSC_SolverComps::makeHinesMatrix(HSCModel *model, HSC_HinexMatrix& matrix)
{
//	unsigned int size = model->compts.size();
//
//	/*
//	 * Some convenience variables
//	 */
//	vector< double > CmByDt;
//	vector< double > Ga;
//	for ( unsigned int i = 0; i < size; i++ ) {
//		CmByDt.push_back( model->compts[ i ].Cm / ( dt / 2.0 ) );
//		Ga.push_back( 2.0 / model->compts[ i ].Ra );
//	}
//
//	/* Each entry in 'coupled' is a list of electrically coupled compartments.
//	 * These compartments could be linked at junctions, or even in linear segments
//	 * of the cell.
//	 */
//	vector< vector< unsigned int > > coupled;
//	for ( unsigned int i = 0; i < model->compts.size(); i++ )
//		if ( model->compts[ i ].children.size() >= 1 ) {
//			coupled.push_back( model->compts[ i ].children );
//			coupled.back().push_back( i );
//		}
//
//	matrix.clear();
//	matrix.resize( size );
//	for ( unsigned int i = 0; i < size; ++i )
//		matrix[ i ].resize( size );

//	// Setting diagonal elements
//	for ( unsigned int i = 0; i < size; i++ )
//		matrix[ i ][ i ] = CmByDt[ i ] + 1.0 / tree[ i ].Rm;
//
//	double gi;
//	vector< vector< unsigned int > >::iterator group;
//	vector< unsigned int >::iterator ic;
//	for ( group = coupled.begin(); group != coupled.end(); ++group ) {
//		double gsum = 0.0;
//
//		for ( ic = group->begin(); ic != group->end(); ++ic )
//			gsum += Ga[ *ic ];
//
//		for ( ic = group->begin(); ic != group->end(); ++ic ) {
//			gi = Ga[ *ic ];
//
//			matrix[ *ic ][ *ic ] += gi * ( 1.0 - gi / gsum );
//		}
//	}
//
//	// Setting off-diagonal elements
//	double gij;
//	vector< unsigned int >::iterator jc;
//	for ( group = coupled.begin(); group != coupled.end(); ++group ) {
//		double gsum = 0.0;
//
//		for ( ic = group->begin(); ic != group->end(); ++ic )
//			gsum += Ga[ *ic ];
//
//		for ( ic = group->begin(); ic != group->end() - 1; ++ic ) {
//			for ( jc = ic + 1; jc != group->end(); ++jc ) {
//				gij = Ga[ *ic ] * Ga[ *jc ] / gsum;
//
//				matrix[ *ic ][ *jc ] = -gij;
//				matrix[ *jc ][ *ic ] = -gij;
//			}
//		}
//	}
}
