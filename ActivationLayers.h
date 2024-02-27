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

__global__ void sig(Tensor32f output, Tensor32f input, bool derv) {
	int id = threadIdx.x + blockDim.x * blockIdx.x;// +blockDim.x * gridDim.x * blockIdx.y;
	if (!derv) {
		output.data[id] = 1 / (1 + exp(-input.data[id]));
	}
	else {
		output.data[id] = input.data[id] * (1 - input.data[id]);
	}
}

__global__ void multiply(Tensor32f output, Tensor32f input) {
	input.data[threadIdx.x + blockDim.x * blockIdx.x] *= output.data[threadIdx.x + blockDim.x * blockIdx.x];
}

class Sigmoid : public Layer {
	void Think() {
		sig <<< outputs->dimentions.batchsize, outputs->dimentions.width >>> (*outputs, *inputs, false);//
	}

	void Learn() {
		sig <<< outputs->dimentions.batchsize, outputs->dimentions.width >>> (*inputs, *inputs, false);//
		sig << < outputs->dimentions.batchsize, outputs->dimentions.width >> > (*inputs, *inputs, true);//
		multiply << < outputs->dimentions.batchsize, outputs->dimentions.width >> > (*outputs, *inputs);//
		//cudaMemcpy(inputs->data, outputs->data, outputs->sizeInBytes, cudaMemcpyDeviceToDevice);
	}

	void CreateOutput() {
		outputs = new Tensor32f(inputs);
	}
};