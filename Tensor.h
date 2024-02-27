#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <Windows.h>

struct dim4 {
	int batchsize = 1;
	int width = 1;
	int height = 1;
	int depth = 1;

	dim4(int Batchsize, int Width, int Height, int Depth) {
		batchsize = Batchsize;
		width = Width;
		height = Height;
		depth = Depth;
	}

	dim4(int Batchsize, int Width, int Height) {
		batchsize = Batchsize;
		width = Width;
		height = Height;
	}

	dim4(int Width, int Height) {
		width = Width;
		height = Height;
	}

	dim4(int Width) {
		width = Width;
	}

	dim4() {};

	int size() {
		return batchsize * width * height * depth;
	}
};

class Tensor32f {
public:
	dim4 dimentions;
	bool onCPU;
	float* data;
	size_t sizeInBytes;

	Tensor32f(dim4 dims, bool onCpu) {
		dimentions = dims;
		onCPU = onCpu;
		sizeInBytes = sizeof(float) * dimentions.size();
		if (onCPU) {
			data = (float*)malloc(sizeInBytes);
		}
		else {
			cudaMalloc(&data, sizeInBytes);
		}
	}

	Tensor32f(Tensor32f* other) {
		dimentions = other->dimentions;
		onCPU = other->onCPU;
		sizeInBytes = other->sizeInBytes;
		if (onCPU) {
			data = (float*)malloc(sizeInBytes);
			cudaMemcpy(data, other->data, sizeInBytes, cudaMemcpyHostToHost);
		}
		else {
			cudaMalloc(&data, sizeInBytes);
			cudaMemcpy(data, other->data, sizeInBytes, cudaMemcpyDeviceToDevice);
		}

	}

	Tensor32f() {
		dimentions = dim4(0, 0, 0, 0);
		data = nullptr;
		onCPU = true;
		sizeInBytes = 0;
	}

	void Flatten() {
		dimentions = dim4(1, dimentions.size(), 1, 1);
	}

	void FreeArray() {
		if (onCPU) {
			free(data);
		}
		else {
			cudaFree(data);
		}
	}

	void ToGPU() {
		if (!onCPU) {
			return;
		}
		float* temp;
		cudaMalloc(&temp, sizeInBytes);
		cudaMemcpy(temp, data, sizeInBytes, cudaMemcpyHostToDevice);
		free(data);
		data = temp;
		onCPU = false;
	}

	void ToCPU() {
		if (onCPU) {
			return;
		}
		float* temp = (float*)malloc(sizeInBytes);
		cudaMemcpy(temp, data, sizeInBytes, cudaMemcpyDeviceToHost);
		cudaFree(data);
		data = temp;
		onCPU = true;
	}

	void Print(int batches) {
		bool c = onCPU;
		if (!c) {
			ToCPU();
		}
		int id = 0;
		for (int i = 0; i < batches; i++){
			for (int j = 0; j < dimentions.width; j++){
				for (int k = 0; k < dimentions.height; k++) {
					for (int l = 0; l < dimentions.depth; l++) {
						std::cout << data[id];
						id++;
					}
					std::cout << ",";
				}
				std::cout << std::endl;
			}
			std::cout << std::endl;
		}
		if (!c) {
			ToGPU();
		}
	}

	void Print(dim4 d) {
		bool c = onCPU;
		if (!c) {
			ToCPU();
		}
		int id = 0;
		for (int i = 0; i < d.batchsize; i++) {
			for (int j = 0; j < d.width; j++) {
				for (int k = 0; k < d.height; k++) {
					for (int l = 0; l < d.depth; l++) {
						std::cout << data[id];
						id++;
					}
					std::cout << ",";
				}
				std::cout << std::endl;
			}
			//std::cout << std::endl;
		}
		if (!c) {
			ToGPU();
		}
	}
};