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
#include "NeuralNetwork.h"

class Error {
public:
	Tensor32f* answers;
	Tensor32f* guesses;
	float* averageError;
	int sampSize;

	Error(int samplesize) {
		averageError = new float[samplesize];
		sampSize = samplesize;
	}

	float FindError(int numOfResults) {
		guesses->ToCPU();
		answers->ToCPU();
		float cumlutiveError = 0;

		for (int i = 0; i < answers->dimentions.size(); i++) {
			if (i < numOfResults) {
				if (i % 10 == 0) {
					std::cout << std::endl;
				}
				std::cout << guesses->data[i] << " ~ " << answers->data[i] << std::endl;
			}
			guesses->data[i] = (answers->data[i] - guesses->data[i]);
			cumlutiveError += abs(guesses->data[i]);
			guesses->data[i] *= abs(guesses->data[i]);
		}

		for (int i = 0; i < answers->dimentions.size(); i++) {
			if (guesses->data[i] > 0) {
				guesses->data[i] *= 10;
			}
		}

		cumlutiveError /= answers->dimentions.size();
		/*for (int i = sampSize - 1; i > 0; i--) {
			averageError[i] = averageError[i - 1];
		}
		averageError[0] = cumlutiveError;
		for (int i = 1; i < sampSize; i++) {
			cumlutiveError += averageError[1];
		}*/
		guesses->ToGPU();
		answers->ToGPU();
		return cumlutiveError;
	}
private:

};