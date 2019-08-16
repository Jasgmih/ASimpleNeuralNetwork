#pragma once
#include <fstream>
#include "Layer.h"


class Calculation
{
public:
	void training();

private:
	void prediction(AllLayers allLayers);
};

