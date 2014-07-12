#pragma once

namespace pn2s {
namespace models {

class ModelStatistic {
public:
	double dt;
	size_t nModels;
	size_t nCompts_per_model;
	size_t nChannels_all;
	ModelStatistic(): dt(1), nModels(0), nCompts_per_model(0), nChannels_all(0) {}
	ModelStatistic(double _dt, size_t _nModel, size_t _nCompt, size_t numberOfChannels):
		dt(_dt), nModels(_nModel), nCompts_per_model(_nCompt), nChannels_all(numberOfChannels){}
	virtual ~ModelStatistic(){}
};

}
}
