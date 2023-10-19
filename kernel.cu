
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "math.h"
#include "stdlib.h"

__global__ void addKernel(char* arr, double LEFT, double BOTTOM, double RIGHT, double TOP, double SIZE, double MaxIters)
{
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind < (int)SIZE * SIZE)
    {
        int x, y, count;
        double zr, zi, betar, betai, a, b;
        double rsquared, isquared, mr, mi, msquared;

        y = ind / SIZE;
        x = ind - SIZE * y;

        zr = 1.0;
        zi = 0.0;
        betar = LEFT + x * (RIGHT - LEFT) / SIZE;
        betai = BOTTOM + y * (TOP - BOTTOM) / SIZE;

        a = 0.5 * (1 - betar);
        b = -0.5 * betai;
        rsquared = zr * zr;
        isquared = zi * zi;

        for (count = 0; rsquared + isquared >= 2.25 / (betar * betar + betai * betai) && count < MaxIters; count++)
        {
            mr = 2 * a - 1 + exp(a * zr - b * zi) * cos(a * zi + b * zr);
            mi = 2 * b + exp(a * zr - b * zi) * sin(a * zi + b * zr);
            msquared = mr * mr + mi * mi;
            zr = 1 - 2 * (a * mr + b * mi) / msquared;
            zi = 2 * (a * mi - b * mr) / msquared;
            rsquared = zr * zr;
            isquared = zi * zi;
        }

        if (rsquared + isquared >= 2.25 / (betar * betar + betai * betai))
            arr[x * int(SIZE) + y] = '*';
        else
            arr[x * int(SIZE) + y] = '.';
    }
}

void cpu(int argc, char* argv[])
{
    int   x, y, count;
    long double zr, zi, betar, betai, a, b;
    long double rsquared, isquared, mr, mi, msquared;

    double LEFT, BOTTOM, RIGHT, TOP, SIZE, MaxIters;

    if (argc == 1)
    {
        LEFT = -10.;
        BOTTOM = -10.;
        RIGHT = 10.;
        TOP = 10.;
        SIZE = 40;
        MaxIters = 80.;
    }
    else
    {
        LEFT = atof(argv[1]);
        BOTTOM = atof(argv[2]);
        RIGHT = atof(argv[3]);
        TOP = atof(argv[4]);
        SIZE = atof(argv[5]);
        MaxIters = atof(argv[6]);
    }

    char* arr = (char*) malloc(SIZE*SIZE*sizeof(char));
    for (int i = 0; i < SIZE * SIZE; i++)
        arr[i] = ' ';

    for (y = 0; y < SIZE; y++)
    {
        for (x = 0; x < SIZE; x++)
        {
            zr = 1.0;
            zi = 0.0;
            betar = LEFT + x * (RIGHT - LEFT) / SIZE;
            betai = BOTTOM + y * (TOP - BOTTOM) / SIZE;

            a = 0.5 * (1 - betar);
            b = -0.5 * betai;
            rsquared = zr * zr;
            isquared = zi * zi;

            for (count = 0; rsquared + isquared >= 2.25 / (betar * betar + betai * betai) && count < MaxIters; count++)
            {
                mr = 2 * a - 1 + exp(a * zr - b * zi) * cos(a * zi + b * zr);
                mi = 2 * b + exp(a * zr - b * zi) * sin(a * zi + b * zr);
                msquared = mr * mr + mi * mi;
                zr = 1 - 2 * (a * mr + b * mi) / msquared;
                zi = 2 * (a * mi - b * mr) / msquared;
                rsquared = zr * zr;
                isquared = zi * zi;
            }

            if (rsquared + isquared >= 2.25 / (betar * betar + betai * betai))
                arr[x * int(SIZE)+y] = '*';
            else
                arr[x * int(SIZE) + y] = '.';
        }
    }
    for (int i = 0; i < SIZE; i++)
    {
        for (int j = 0; j < SIZE; j++)
            printf("%c", arr[j * int(SIZE) + i]);
        printf("\n");
    }
}

void gpu(int argc, char* argv[])
{
    int   x, y, count;
    long double zr, zi, betar, betai, a, b;
    long double rsquared, isquared, mr, mi, msquared;

    double LEFT, BOTTOM, RIGHT, TOP, SIZE, MaxIters;

    if (argc == 1)
    {
        LEFT = -10.;
        BOTTOM = -10.;
        RIGHT = 10.;
        TOP = 10.;
        SIZE = 40;
        MaxIters = 80.;
    }
    else
    {
        LEFT = atof(argv[1]);
        BOTTOM = atof(argv[2]);
        RIGHT = atof(argv[3]);
        TOP = atof(argv[4]);
        SIZE = atof(argv[5]);
        MaxIters = atof(argv[6]);
    }

    char* arr = (char*)malloc(SIZE * SIZE * sizeof(char));
    for (int i = 0; i < SIZE * SIZE; i++)
        arr[i] = ' ';

    char* dev_a = 0;
  
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, SIZE*SIZE * sizeof(char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, arr, SIZE*SIZE * sizeof(char), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    dim3 threads = 500;
    dim3 blocks = (SIZE * SIZE) / threads.x + 1;
    addKernel << <blocks, threads>> > (dev_a, LEFT, BOTTOM, RIGHT, TOP, SIZE, MaxIters);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(arr, dev_a, SIZE*SIZE * sizeof(char), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_a);

    cudaStatus = cudaDeviceReset();
 
    for (int i = 0; i < SIZE; i++)
    {
        for (int j = 0; j < SIZE; j++)
            printf("%c", arr[j * int(SIZE) + i]);
        printf("\n");
    }
}

int main(int argc, char* argv[])
{
    //cpu(argc, argv);
    gpu(argc, argv);
    return 0;
}