
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <Windows.h>

#include "FileReader.h"
#include "NeuralNetwork.h"
#include "DenseLayer.h"
#include "ActivationLayers.h"
#include "Format.h"

/*int main() {
    Tensor32f input = Tensor32f(dim4(1, 1, 1, 1), true);
    input.data[0] = 1;
    //input.data[1] = 0;
    //input.data[2] = 0;
    //input.data[3] = 1;
    input.ToGPU();
    Tensor32f output = Tensor32f(dim4(1, 1, 1, 1), true);
    output.data[0] = -1;
    //output.data[1] = 1;
    //output.data[2] = 1;
    //output.data[3] = -1;
    output.ToGPU();
    FullyConnectedDense w = FullyConnectedDense(1, 1, .5, 1);
    w.inputs = &input;
    w.outputs = &output;
    w.weights.Print(1);
    w.Learn();
    w.weights.Print(1);
    input.Print(1);
}*/

int main() {
    srand(0);
    ImageDataset Letters = ImageDataset("C:/users/wills/Downloads/Letters.bmp", 0);
    ImageDataset Numbers = ImageDataset("C:/users/wills/Downloads/Numbers.bmp", 0);

    //base Letter AI
    NeuralNetwork baseNN = NeuralNetwork(new ImageLoader(&Letters, 26), 5);
    Flatten* bf1 = (Flatten*)                           baseNN.AddLayer(new Flatten());
    FullyConnectedDense* bw1 = (FullyConnectedDense*)   baseNN.AddLayer(new FullyConnectedDense(1024, 256, 2, .5));
    Sigmoid* bs1 = (Sigmoid*)                           baseNN.AddLayer(new Sigmoid());
    FullyConnectedDense* bw2 = (FullyConnectedDense*)   baseNN.AddLayer(new FullyConnectedDense(256, 32, 2, .5));
    Sigmoid* bs2 = (Sigmoid*)                           baseNN.AddLayer(new Sigmoid());
    FullyConnectedDense* bw3 = (FullyConnectedDense*)   baseNN.AddLayer(new FullyConnectedDense(32, 26, 2, .5));
    Sigmoid* bs3 = (Sigmoid*)                           baseNN.AddLayer(new Sigmoid());

    //expanding Number Ai
    NeuralNetwork expNN = NeuralNetwork(new ImageLoader(&Numbers,10), 5);
    Flatten* pf1 = (Flatten*)expNN.AddLayer(new Flatten());
    FullyConnectedDense* pw1 = (FullyConnectedDense*)   expNN.AddLayer(new FullyConnectedDense(1024, 256, 2, .5));
    Sigmoid* ps1 = (Sigmoid*)                           expNN.AddLayer(new Sigmoid());
    FullyConnectedDense* pw2 = (FullyConnectedDense*)   expNN.AddLayer(new FullyConnectedDense(256, 32, 2, .5));
    Sigmoid* ps2 = (Sigmoid*)                           expNN.AddLayer(new Sigmoid());
    FullyConnectedDense* pw3 = (FullyConnectedDense*)   expNN.AddLayer(new FullyConnectedDense(32, 10, 2, .5));
    Sigmoid* ps3 = (Sigmoid*)                           expNN.AddLayer(new Sigmoid());

    //extending Number Ai
    NeuralNetwork extNN = NeuralNetwork(new ImageLoader(&Numbers,10), 5);
    Flatten* tf1 = (Flatten*)                           extNN.AddLayer(new Flatten());
    FullyConnectedDense* tw1 = (FullyConnectedDense*)   extNN.AddLayer(new FullyConnectedDense(1024, 256, 2, .5));
    Sigmoid* ts1 = (Sigmoid*)                           extNN.AddLayer(new Sigmoid());
    FullyConnectedDense* tw2 = (FullyConnectedDense*)   extNN.AddLayer(new FullyConnectedDense(256, 32, 2, .5));
    Sigmoid* ts2 = (Sigmoid*)                           extNN.AddLayer(new Sigmoid());
    FullyConnectedDense* tw3 = (FullyConnectedDense*)   extNN.AddLayer(new FullyConnectedDense(32, 26, 2, .5));
    Sigmoid* ts3 = (Sigmoid*)                           extNN.AddLayer(new Sigmoid());
    FullyConnectedDense* tw4 = (FullyConnectedDense*)   extNN.AddLayer(new FullyConnectedDense(26, 10, 2, .5));
    Sigmoid* ts4 = (Sigmoid*)                           extNN.AddLayer(new Sigmoid());

    baseNN.createBatches(256);
    expNN.createBatches(256);
    extNN.createBatches(256);
    while(true){
        //base training
        bw1->GenerateWeights(2);
        bw2->GenerateWeights(2);
        bw3->GenerateWeights(2);
        for (int j = 0; j < 250; j++) {
            baseNN.Learn(0);
        }
        std::cout<< " Base network error - " << baseNN.Learn(0) << std::endl;;
        for (int k = 0; k < 10; k++) {
            //expanding
            pw1->weights.FreeArray();
            pw1->weights = Tensor32f(&(bw1->weights));
            pw2->weights.FreeArray();
            pw2->weights = Tensor32f(&(bw2->weights));
            pw3->GenerateWeights(2);
            int counter = 0;
            float error = 1;
            while(counter < 10000){
                counter++;
                error = expNN.Learn(0);
            }
            std::cout << "expanding error " << k << " - " << error << std::endl;

            tw1->weights.FreeArray();
            tw1->weights = Tensor32f(&(bw1->weights));
            tw2->weights.FreeArray();
            tw2->weights = Tensor32f(&(bw2->weights));
            tw3->weights.FreeArray();
            tw3->weights = Tensor32f(&(bw3->weights));
            tw3->GenerateWeights(2);
            counter = 0;
            error = 1;
            while (counter < 10000) {
                counter++;
                error = extNN.Learn(0);
            }
            std::cout << "extending error " << k << " - " << error << std::endl;
        }
    }
    return 69;
}

