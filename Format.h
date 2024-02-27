#pragma once
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

class Flatten : public Layer{

	virtual void Think() {
		cudaMemcpy(outputs->data, inputs->data, inputs->sizeInBytes, cudaMemcpyDeviceToDevice);
	}

	virtual void Learn() {
		cudaMemcpy(inputs->data, outputs->data, inputs->sizeInBytes, cudaMemcpyDeviceToDevice);
	}

	virtual void CreateOutput() {
		outputs = new Tensor32f(dim4(inputs->dimentions.batchsize, inputs->dimentions.size() / inputs->dimentions.batchsize, 1),false);
	}
};