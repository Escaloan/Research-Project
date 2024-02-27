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
#include "FileReader.h"

class Loader {
public:

	Tensor32f* output;
	Tensor32f* answers;

	virtual void CreateBatch(int BatchSize) { return; }
	virtual void LoadTesting() { return; }
	virtual void LoadTraining() { return; }
};

class TestLoader: public Loader {
public:
	float source[12];
	//float correct[4];
	void CreateBatch(int BatchSize) {
		output = new Tensor32f(dim4(4, 3, 1, 1), false);
		answers = new Tensor32f(dim4(4, 3, 1, 1), true);

		source[0] = 0; source[1] = 0; source[2] = 1;// correct[0] = 1;
		source[3] = 1; source[4] = 1; source[5] = 1;// correct[1] = 0;
		source[6] = 1; source[7] = 0; source[8] = 1;// correct[2] = 0;
		source[9] = 0; source[10] = 1; source[11] = 1;// correct[3] = 0;
	}

	void LoadTraining() {
		cudaMemcpy(output->data, source, 12 * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(answers->data, source, 12 * sizeof(float), cudaMemcpyHostToHost);
	}

	void LoadTesting() {
		cudaMemcpy(output->data, source, 12 * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(answers->data, source, 12 * sizeof(float), cudaMemcpyHostToHost);
	}
};

__global__ void GenerateAnswers(Tensor32f answers, Image* images, int randomOff, int setSize, int n) {
	int batch = threadIdx.x + blockDim.x * blockIdx.x;
	if (batch >= n) {
		return;
	}
	for (int i = 0; i < 10; i++) {
		answers.data[batch * 10 + i] = 0;
		if (images[(randomOff + batch) % setSize].catagory == i) {
			answers.data[batch * 10 + i] = 1;
		}
	}
}


class ImageLoader : public Loader {
public:
	ImageLoader(ImageDataset* data, int outputsize) {
		images = *data;
		batchOffset = 0;
		output = nullptr;
		answers = nullptr;
		outsize = outputsize;
	}

	ImageLoader(ImageLoader* other) {
		images = other->images;
		batchOffset = other->batchOffset;
		batchSize = other->batchSize;
	}

	void CreateBatch(int BatchSize) {
		if (output != nullptr) {
			output->FreeArray();
			free(output);
			answers->FreeArray();
			free(answers);
		}
		output = new Tensor32f(dim4(BatchSize, 32, 32, 1), false);
		answers = new Tensor32f(dim4(BatchSize, outsize, 1, 1), false);
		batchSize = BatchSize;
	}

	void LoadTraining() {
		for (int i = 0; i < batchSize; i++) {
			cudaMemcpy(&output->data[1024 * i], images.imageSet[(batchOffset + i) % images.trainingSetSize].image, sizeof(float) * 1024, cudaMemcpyDeviceToDevice);
			//cudaMemcpy(&answers[i], &images.imageSet[(batchOffset + i) % images.trainingSetSize].catagory, sizeof(int), cudaMemcpyDeviceToDevice);
		}
		GenerateAnswers <<<batchSize / 32 + 1, 32 >>> (*answers, images.imageSet, batchOffset, images.trainingSetSize, batchSize);//
		//batchOffset += batchSize;
		batchOffset = batchOffset % images.trainingSetSize;
	}

	void LoadTesting() {
		int randomOff = rand() % images.testingSetSize;
		for (int i = 0; i < batchSize; i++) {
			cudaMemcpy(&output->data[1024 * i], images.testingSet[(randomOff + i) % images.testingSetSize].image, sizeof(float) * 1024, cudaMemcpyDeviceToDevice);
		}
		GenerateAnswers <<<batchSize / 32 + 1,32 >>> (*answers, images.testingSet, randomOff, images.testingSetSize, batchSize);//
	}

	ImageDataset images;
	int batchOffset;
	int batchSize;

private:
	int outsize = 0;
};