#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <Windows.h>
#include <bitset>
/*
* This should be rewritten to support other data types.
*/

struct BitMapFileHeader {
    WORD bfType;  //specifies the file type
    DWORD bfSize;  //specifies the size in bytes of the bitmap file
    WORD bfReserved1;  //reserved; must be 0
    WORD bfReserved2;  //reserved; must be 0
    DWORD bfOffBits;  //specifies the offset in bytes from the bitmapfileheader to the bitmap bits
};

struct BitMapInfoHeader{
    LONG biSize;  //specifies the number of bytes required by the struct
    LONG  biWidth;  //specifies width in pixels
    LONG  biHeight;  //specifies height in pixels
    WORD  biPlanes;  //specifies the number of color planes, must be 1
    WORD  biBitCount;  //specifies the number of bits per pixel
    LONG biCompression;  //specifies the type of compression
    LONG biSizeImage;  //size of image in bytes
    LONG  biXPelsPerMeter;  //number of pixels per meter in x axis
    LONG  biYPelsPerMeter;  //number of pixels per meter in y axis
    LONG biClrUsed;  //number of colors used by the bitmap
    LONG biClrImportant;  //number of colors that are important
};

struct BMP {
    BitMapFileHeader fileHeader;
    BitMapInfoHeader infoHeader;
    BYTE*            rawImage;
};

struct Image {
    float image[1024];
    int catagory;

    float Get(int x, int y) {
        return image[x + 32 * y];
    }

    void Set(int x, int y, float val) {
        image[x + 32 * y] = val;
    }
};

void printImage(Image* gpuImage, float limit) {
    Image cImage;
    cudaMemcpy(&cImage, gpuImage, sizeof(Image), cudaMemcpyDeviceToHost);
    int index = 0;
    std::cout << std::endl;
    for (int i = 0; i < 32; i++) {
        for (int j = 0; j < 32; j++) {
            if (cImage.image[index] > limit) {
                std::cout << "#";
            }
            else {
                std::cout << " ";
            }
            index++;
        }
        std::cout << std::endl;
    }
    std::cout << cImage.catagory;
}

/* Transfer the raw image into a dataset
*  t.x - width
*  t.y - height
*  b.x - set
*  b.y - cata
*/
__global__ void FillImage(Image* images, BMP bmpImage) {
    int imageIdx = blockIdx.x + blockIdx.y * gridDim.x;
    int rawIdx = blockIdx.x * 33
        + threadIdx.x
        + (gridDim.x * 33)
        * (blockIdx.y * 33 + threadIdx.y);
    images[imageIdx].image[threadIdx.x + 32 * (31 - threadIdx.y)] = (float)((int)bmpImage.rawImage[rawIdx * 3]) / 256.0f;
    return;
}

//
__global__ void GenerateCatagories(Image* images) {
    images[threadIdx.x + blockIdx.x * blockDim.x].catagory = blockIdx.x;
}

class ImageDataset {
public:
    ImageDataset(char* filename, int reseveForTesting) {
        testingSetSize = reseveForTesting;
        ReadFile(filename);
        AllocateSet();
        RenderSet();
        ScrambleSet();
        testingSet = &imageSet[trainingSetSize];
    }

    ImageDataset(){}

    Image* imageSet;
    Image* testingSet;
    int trainingSetSize;
    int testingSetSize;

private:
    //reads the BMP file
    void ReadFile(char* filename) {
        FILE* filePtr;
        filePtr = fopen(filename, "rb");
        if (filePtr == NULL) {
            printf("error");
            abort();
        }

        fread(&rawFile.fileHeader.bfType, sizeof(WORD), 1, filePtr);
        fread(&rawFile.fileHeader.bfSize, sizeof(DWORD), 1, filePtr);
        fread(&rawFile.fileHeader.bfReserved1, sizeof(WORD), 1, filePtr);
        fread(&rawFile.fileHeader.bfReserved2, sizeof(WORD), 1, filePtr);
        fread(&rawFile.fileHeader.bfOffBits, sizeof(DWORD), 1, filePtr);

        fread(&(rawFile.infoHeader.biSize), sizeof(DWORD), 1, filePtr);
        fread(&rawFile.infoHeader.biWidth, sizeof(LONG), 1, filePtr);
        fread(&rawFile.infoHeader.biHeight, sizeof(LONG), 1, filePtr);
        fread(&rawFile.infoHeader.biPlanes, sizeof(WORD), 1, filePtr);
        fread(&rawFile.infoHeader.biBitCount, sizeof(WORD), 1, filePtr);
        fread(&rawFile.infoHeader.biCompression, sizeof(DWORD), 1, filePtr);
        fread(&rawFile.infoHeader.biSizeImage, sizeof(DWORD), 1, filePtr);
        fread(&rawFile.infoHeader.biXPelsPerMeter, sizeof(LONG), 1, filePtr);
        fread(&rawFile.infoHeader.biYPelsPerMeter, sizeof(LONG), 1, filePtr);
        fread(&rawFile.infoHeader.biClrUsed, sizeof(DWORD), 1, filePtr);
        fread(&rawFile.infoHeader.biClrImportant, sizeof(DWORD), 1, filePtr);
        rawFile.infoHeader.biSizeImage = rawFile.infoHeader.biWidth * rawFile.infoHeader.biHeight * 3;
        if (rawFile.fileHeader.bfType != 0x4D42) {
            printf("error");
        }
        cudaMallocManaged(&rawFile.rawImage, rawFile.infoHeader.biSizeImage);
        fread(rawFile.rawImage, rawFile.infoHeader.biSizeImage, 1, filePtr);
        fclose(filePtr);
    }

    void AllocateSet() {
        totalSetSize = (rawFile.infoHeader.biWidth / 33) * (rawFile.infoHeader.biHeight / 33);
        trainingSetSize = totalSetSize - testingSetSize;
        cudaMalloc(&imageSet, sizeof(Image) * totalSetSize);
    }

    void RenderSet() {
        dim3 threads = dim3(32, 32, 1);
        dim3 blocks = dim3(rawFile.infoHeader.biWidth / 33, rawFile.infoHeader.biHeight / 33, 1);
        cudaDeviceSynchronize();
        FillImage <<< blocks, threads >>> (imageSet, rawFile);//
        GenerateCatagories <<< blocks.y, blocks.x >>> (imageSet);//
        cudaDeviceSynchronize();
    }

    void ScrambleSet() {
        Image placeholder;
        for (int i = 0; i < totalSetSize; i++) {
            int other = rand() % totalSetSize;
            cudaMemcpy(&placeholder, &(imageSet[i]), sizeof(Image), cudaMemcpyDeviceToHost);
            cudaMemcpy(&(imageSet[i]), &(imageSet[other]), sizeof(Image), cudaMemcpyDeviceToDevice);
            cudaMemcpy(&(imageSet[other]), &placeholder, sizeof(Image), cudaMemcpyHostToDevice);
        }
    }
    
    BMP rawFile;
    int totalSetSize;
};