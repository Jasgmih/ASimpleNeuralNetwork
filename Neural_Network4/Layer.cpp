#include "Layer.h"


Layer::Layer(int number_LastLayer, int number_ThisLayer)
{
	this->number_ThisLayer = number_ThisLayer;
	this->number_LastLayer = number_LastLayer;
}


void Layer::init()
{
	this->activation.resize(this->number_ThisLayer);

	vector <double> w;
	for (int i = 0; i < number_LastLayer; i++) {
		w.emplace_back(0.);
	}

	for (int i = 0; i < number_ThisLayer; i++) {
		weight.emplace_back(w);
		biase.emplace_back(0.);
	}

}

void AllLayers::init()
{
	
	int num;
	cout << "Please input the number of neural units in every layer: " << endl;
	for (int i = 0; i < LAYER; i++) {
		cout << "layer " << i << ": " << endl;
		cin >> num;
		numberLayer[i] = num;
		if (i == 0) Layers.emplace_back(0, numberLayer[i]);
		else Layers.emplace_back(numberLayer[i-1], numberLayer[i]);
	}
	
	vector <double> row;
	row.resize(numberLayer[0]);
	for (int i = 0; i < NUMBER_DATA; i++) {
		input.emplace_back(row);
		output.emplace_back(row);
		for (int j = 0; j < numberLayer[0]; j++) {
			if (i == j) {
				input[i][j] = 1;
				output[i][j] = 1;
			}
			else {
				input[i][j] = 0;
				output[i][j] = 0;
			}
		}
	}

	for (int layer = 0; layer < LAYER; layer++) {
		Layers[layer].init();
	}

	double single_a;
	double single_z = 0.;
	vector <double> a_unit;
	vector <vector <double>> a_layer;

	for (int i = 0; i < NUMBER_DATA; i++) {

		for (int j = 0; j < numberLayer[0]; j++) {
			a_unit.emplace_back(input[i][j]);
		}
		a_layer.emplace_back(a_unit);
		a_unit.clear();

		for (int layer = 1; layer < LAYER; layer++) {

			for (int j = 0; j < numberLayer[layer]; j++) {
				
				for (int row = 0; row < numberLayer[layer - 1]; row++) {
					single_z += Layers[layer].weight[j][row] * a_layer[layer - 1][row];
				}
				single_a = g(single_z);
				a_unit.emplace_back(single_a);
			}
			a_layer.emplace_back(a_unit);
			a_unit.clear();
		}

		a.emplace_back(a_layer);
		a_layer.clear();
	}


}


double AllLayers::g(double z)
{
	return 1. / (1. + exp(-z));
}

double AllLayers::z(int layer, int j, int i)
{
	double z = 0.;

	for (int row = 0; row < numberLayer[layer - 1]; row++) {
		z += Layers[layer].weight[j][row] * a[i][layer-1][row];
	}
	z += Layers[layer].biase[j];
	return z;
}

double AllLayers::derivative_g(double z)
{
	return g(z)*(1. - g(z));
}



double AllLayers::partial_weight(int layer, int j, int k,int i)
{
	return a[i][layer-1][k] * derivative_g(z(layer,j,i))*partial_activation(layer,j,i);
}

double AllLayers::partial_activation(int layer, int j, int i)
{
	double sum = 0.;
	if (layer == LAYER-1) {
		return 2.*(a[i][layer][j] - output[i][j]);
	}
	for (int nextLayerRow = 0; nextLayerRow < numberLayer[layer + 1]; nextLayerRow++) {
		sum += Layers[layer + 1].weight[nextLayerRow][j] 
			* derivative_g(z(layer + 1, nextLayerRow, i))
			*partial_activation(layer + 1, nextLayerRow,i);
	}
	return sum;
}

double AllLayers::partial_biase(int layer, int j, int i)
{
	return derivative_g(z(layer, j,i))*partial_activation(layer,j,i);
}

double AllLayers::a_single(int layer, int j, int i)
{
	if (layer == 0) return input[i][j];
	else return g(z(layer, j, i));
}

void AllLayers::a_recalculation()
{
	for (int i = 0; i < NUMBER_DATA; i++) {
		for (int layer = 1; layer < LAYER; layer++) {
			for (int j = 0; j < numberLayer[layer]; j++) {
				a[i][layer][j] = a_single(layer, j, i);
			}
		}
	}
}

