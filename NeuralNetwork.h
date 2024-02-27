#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <Windows.h>
#include "Tensor.h"
#include "Loader.h"
#include "ErrorCalculator.h"

class Layer {
public:
	Tensor32f* inputs;
	Tensor32f* outputs;
	
	virtual void Think() {return;}
	virtual void Learn() {return;}
	virtual void CreateOutput() {return;}
};

class NeuralNetwork {
public:
	NeuralNetwork* self;
	Loader* head;
	Error* error;
	Layer* layers[256];

	NeuralNetwork(Loader* loader, int errorSampleSize){
		self = (NeuralNetwork*)&self;
		NumberOfLayers = 0;
		head = loader;
		error = new Error(errorSampleSize);
	}

	void createBatches(int batchsize) {
		head->CreateBatch(batchsize);
		layers[0]->inputs = head->output;
		for (int i = 0; i < NumberOfLayers - 1; i++) {
			layers[i]->CreateOutput();
			layers[i + 1]->inputs = layers[i]->outputs;
		}
		layers[NumberOfLayers - 1]->CreateOutput();
		error->guesses = layers[NumberOfLayers - 1]->outputs;
		error->answers = head->answers;
	}

	Layer* AddLayer(Layer* layer) {
		layers[NumberOfLayers] = layer;
		NumberOfLayers++;
		return layer;
	}

	float Learn(int results) {
		head->LoadTraining();
		for (int i = 0; i < NumberOfLayers; i++) {
			layers[i]->Think();
			if (i > 0) {
				//std::cout << i << std::endl;
				//layers[i]->outputs->Print(1);
			}
			cudaThreadSynchronize();
		}
		float err = error->FindError(results);
		for (int i = NumberOfLayers - 1; i >= 0; i--) {
			layers[i]->Learn();
		}
		return err;
	}

	float Test() {
		head->LoadTesting();
		for (int i = 0; i < NumberOfLayers; i++) {
			layers[i]->Think();
			cudaThreadSynchronize();
		}
		float err = error->FindError(0);
		return err;
	}

private:
	int NumberOfLayers;
};