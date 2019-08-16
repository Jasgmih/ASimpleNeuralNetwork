#pragma once
#include <iostream>
#include <vector>
using namespace std;

#define LAYER 3
#define NUMBER_DATA 8
#define NUMBER_ITERATION 100
#define ALPHA 0.0001
#define LAMBDA 0.1

class Layer
{
public:
	Layer(int number_ThisLayer,int number_LastLayer);
	void init();

public:
	vector <double> activation;
	vector <vector<double>> weight;
	vector <double> biase;

private:
	int number_ThisLayer;
	int number_LastLayer;
	
};

class AllLayers {
public:
	void init();
	double partial_weight(int layer, int j, int k,int i);
	double partial_activation(int layer, int j, int i);
	double partial_biase(int layer, int j, int i);
	void a_recalculation();

public:
	vector <Layer> Layers;
	int numberLayer[LAYER];
	vector < vector <double>> input;
	vector < vector <double>> output;
	vector <vector < vector <double>>>  a;

private:
	double a_single(int layer, int j, int i);
	double g(double z);
	double z(int layer, int j, int i);
	double derivative_g(double z);
	

};

