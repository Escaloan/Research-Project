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
__device__ float sign(float x) {
	if (x < 0) {
		return -1;
	}
	if (x > 0) {
		return 1;
	}
	return 0;
}

__global__ void LinearThink(Tensor32f in, Tensor32f out, Tensor32f weights) {
	float sum = 0;
	int weightHeader = threadIdx.x * in.dimentions.width;
	int inputHeader = blockIdx.x * in.dimentions.width;
	for (int i = 0; i < in.dimentions.width; i++) {
		sum += weights.data[weightHeader + i] * in.data[inputHeader + i];
	}
	out.data[threadIdx.x + blockIdx.x * blockDim.x] = sum;
}

__global__ void BackPropWeights(Tensor32f in, Tensor32f out, Tensor32f weights, float learnRate) {
	float deltaW = 0;
	for (int i = 0; i < in.dimentions.batchsize; i++) {
		deltaW += in.data[i * in.dimentions.width + blockIdx.x] * out.data[i * out.dimentions.width + threadIdx.x];
	}
	weights.data[threadIdx.x * gridDim.x + blockIdx.x] += deltaW * learnRate / (float)in.dimentions.batchsize;
}

__global__ void BackPropError(Tensor32f in, Tensor32f out, Tensor32f weights) {
	float sum = 0;
	int outHeader = blockIdx.x * out.dimentions.width;
	for (int i = 0; i < out.dimentions.width; i++) {
		sum += sign(weights.data[threadIdx.x + i * out.dimentions.width]) * out.data[outHeader + i];
	}
	in.data[threadIdx.x + blockIdx.x * blockDim.x] = sum / out.dimentions.width;
}

class FullyConnectedDense : public Layer {
public:
	Tensor32f weights;

	FullyConnectedDense(int inputsize, int outputsize, float generationRange, float learnRate) {
		weights = Tensor32f(dim4(outputsize, inputsize), true);
		learningRate = learnRate;
		GenerateWeights(generationRange);
	}

	void GenerateWeights(float range = 1) {
		weights.ToCPU();
		for (int i = 0; i < weights.dimentions.size(); i++) {
			weights.data[i] = (((float)(rand() % RAND_MAX) / RAND_MAX * 2) - 1) * range;
		}
		weights.ToGPU();
	}

	void CreateOutput() {
		outputs = new Tensor32f(dim4(inputs->dimentions.batchsize, weights.dimentions.width, 1, 1), false);
	}

	void Think() {
		LinearThink <<< inputs->dimentions.batchsize, weights.dimentions.width >>> (*inputs, *outputs, weights);//
		return;
	}

	void Learn() {
		//outputs->Print(4);
		BackPropWeights <<< weights.dimentions.height, weights.dimentions.width>>> (*inputs, *outputs, weights, learningRate);//
		cudaDeviceSynchronize();
		BackPropError <<<inputs->dimentions.batchsize, weights.dimentions.height>>> (*inputs, *outputs, weights);//
		cudaDeviceSynchronize();
		//weights.Print(1);
	}

private:
	float learningRate;
};

