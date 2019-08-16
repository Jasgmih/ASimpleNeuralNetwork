#include "Calculation.h"

void Calculation::training()
{
	AllLayers allLayers;
	allLayers.init();
	double sum_partial_weight = 0.;
	double sum_partial_biase = 0.;
	vector <double> a_unit;
	vector <vector <double>> a_layer;
	double cost = 0.;


	ofstream outputfile;
	outputfile.open("myfile.txt");
	if (outputfile.fail()) {
		perror("myfile.txt");
		return;
	}


	for (int iteration = 0; iteration < NUMBER_ITERATION; iteration++) {
		cout <<"iteration : " <<iteration << endl;

		/////////output to file///////////////////
		
		outputfile <<endl << endl << endl << "iteration : " << iteration << endl;
		outputfile << "w" << endl;

		for (int l = 1; l < LAYER; l++) {
			outputfile << "layer: " << l << endl;
			for (int j = 0; j < allLayers.numberLayer[l]; j++) {
				for (int k = 0; k < allLayers.numberLayer[l - 1]; k++) {
					outputfile << allLayers.Layers[l].weight[j][k] << "	";
				}
				outputfile << endl;
			}
			outputfile << endl;
		}
		outputfile << "a" << endl;
		for (int i = 0; i < NUMBER_DATA; i++) {
			outputfile << "i=" << i << endl;

			for (int l = 0; l < LAYER; l++) {
				outputfile << "layer =" << l << endl;
				for (int j = 0; j < allLayers.numberLayer[l]; j++) {
					outputfile << allLayers.a[i][l][j] << "	";

				}
				outputfile << endl;
			}
			outputfile << endl;
		}

		outputfile << "b" << endl;

		for (int l = 1; l < LAYER; l++) {
			outputfile << "layer: " << l << endl;
			for (int j = 0; j < allLayers.numberLayer[l]; j++) {
				outputfile << allLayers.Layers[l].biase[j] << "	";
			}
			outputfile << endl;
		}
		
		for (int i = 0; i < NUMBER_DATA; i++) {
			for (int j = 0; j < allLayers.numberLayer[LAYER - 1]; j++) {
				cost += pow (allLayers.a[i][LAYER - 1][j] - allLayers.output[i][j],2);
			}

		}
		outputfile << "cost: " << cost << endl;
	

		for (int l = LAYER - 1; l > 0; l--) {
			for (int j = 0; j < allLayers.numberLayer[l]; j++) {
				for (int k = 0; k < allLayers.numberLayer[l - 1]; k++) {
					for (int i = 0; i < NUMBER_DATA; i++) {
						sum_partial_weight += allLayers.partial_weight(l, j, k, i);
					}
						allLayers.Layers[l].weight[j][k] -= ALPHA * (sum_partial_weight/ NUMBER_DATA + 
							LAMBDA* allLayers.Layers[l].weight[j][k]);
				}
			}
		}

		for (int l = LAYER - 1; l > 0; l--) {

			for (int j = 0; j < allLayers.numberLayer[l]; j++) {
				for (int i = 0; i < NUMBER_DATA; i++) {
					sum_partial_biase += allLayers.partial_biase(l, j, i);
				}
				allLayers.Layers[l].biase[j] -= ALPHA * sum_partial_biase / NUMBER_DATA;
			}
		}
		allLayers.a_recalculation();
	}
	cout << "Training is finished!" << endl;

	outputfile.close();


	prediction(allLayers);

}

void Calculation::prediction(AllLayers allLayers)
{
	vector <double> input;
	vector <double> output;
	vector <double> a_prediction;
	vector <double> a_prediction_next;
	double z = 0.;

	input.resize(allLayers.numberLayer[0]);
	for (int i = 0; i < allLayers.numberLayer[0]; i++) {
		if (i == 0) input[i] = 1.;
		else input[i] = 0.;
	}
	/*input[0] = 1.;
	input[1] = 0.;
	input[2] = 0.;
	input[3] = 0.;
	input[4] = 0.;
	input[5] = 0.;
	input[6] = 0.;
	input[7] = 0.;*/

	for (int j = 0; j < allLayers.numberLayer[0]; j++) {
		a_prediction.emplace_back(input[j]);
	}

	for (int layer = 1; layer < LAYER; layer++) {
		
		for (int j = 0; j < allLayers.numberLayer[layer]; j++) {
			
			for (int k = 0; k < allLayers.numberLayer[layer - 1]; k++) {
				z += allLayers.Layers[layer].weight[j][k] * a_prediction[k];
			}
			z += allLayers.Layers[layer].biase[j];

			a_prediction_next.emplace_back(1./1.+ exp(z));
		}
		a_prediction.clear();
		for (int j = 0; j < allLayers.numberLayer[layer]; j++) {
			a_prediction.emplace_back(a_prediction_next[j]);
		}
		a_prediction_next.clear();


	}
	cout << "the output is: " << endl;

	ofstream outputfile;
	outputfile.open("myoutput.txt");
	if (outputfile.fail()) {
		perror("myoutput.txt");
		return;
	}


	for (int j = 0; j < allLayers.numberLayer[LAYER - 1]; j++) {
		output.emplace_back(a_prediction[j]);
		cout << output[j] << endl;
		outputfile << output[j] << endl;
	}

	
	
	outputfile.close();




}

