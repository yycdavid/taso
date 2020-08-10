/* Copyright 2018 Stanford
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "taso/ops.h"
#include "taso/cuda_helper.h"
using namespace taso;

Model::Model()
: isTraining(false), print_cost(false)
{
  //int* a = (int*) malloc(sizeof(int) * 8);
  checkCUDA(cudaSetDevice(0));
  checkCUDNN(cudnnCreate(&dnn));
  checkCUDA(cublasCreate(&blas));
  workSpaceSize = WORK_SPACE_SIZE;
  global_unique_id = 100;
  checkCUDA(cudaMalloc(&workSpace, workSpaceSize));
  // printf("handle.workSpace = 0x%x\n", workSpace);
  // create all descriptors
  checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&biasTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&scaleTensor));
  checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
  checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
  checkCUDNN(cudnnCreatePoolingDescriptor(&poolDesc));
  checkCUDNN(cudnnCreateActivationDescriptor(&actiDesc));
  checkCUDNN(cudnnCreateOpTensorDescriptor(&opDesc));
  // allocate tensors for measuring performance
  checkCUDA(cudaMalloc(&inputPtr, MAX_TENSOR_SIZE));
  checkCUDA(cudaMalloc(&biasPtr, MAX_TENSOR_SIZE));
  checkCUDA(cudaMalloc(&outputPtr, MAX_TENSOR_SIZE));
  checkCUDA(cudaMalloc(&filterPtr, MAX_TENSOR_SIZE));
  // create tensors for batch norm
  checkCUDA(cudaMalloc(&scalePtr, MAX_TENSOR_SIZE));
  checkCUDA(cudaMalloc(&runningMean, MAX_TENSOR_SIZE));
  checkCUDA(cudaMalloc(&runningVar, MAX_TENSOR_SIZE));
  checkCUDA(cudaMalloc(&saveMean, MAX_TENSOR_SIZE));
  checkCUDA(cudaMalloc(&saveVar, MAX_TENSOR_SIZE));
  // create cuda events
  checkCUDA(cudaEventCreate(&startEvent));
  checkCUDA(cudaEventCreate(&endEvent));
}

float Model::measure_oplist_runtime(const std::vector<OpBase*>& opBaseList)
{
  // warmup
  for (int times = 0; times < 100; times++)
    for (int i = 0; i < opBaseList.size(); i++)
      opBaseList[i]->forward();

  // Estimate time per pass
  float time_taken;
  checkCUDA(cudaDeviceSynchronize());
  checkCUDA(cudaEventRecord(startEvent));
  for (int times = 0; times < 10; times++)
    for (int i = 0; i < opBaseList.size(); i++)
      opBaseList[i]->forward();
  checkCUDA(cudaEventRecord(endEvent));
  checkCUDA(cudaEventSynchronize(endEvent));
  cudaEventElapsedTime(&time_taken, startEvent, endEvent);
  float time_per_pass = time_taken / 10.0;

  int num_minutes = 2;
  int num_passes = num_minutes * 60 * 1000 / time_per_pass;

  // Run GPU to stable temperature
  for (int times = 0; times < num_passes; times++)
    for (int i = 0; i < opBaseList.size(); i++)
      opBaseList[i]->forward();

  const int num_runs = 500;
  
  // measure runtime
  float total_time = 0;
  for (int times = 0; times < num_runs; times++) {
    float milliseconds;
    checkCUDA(cudaDeviceSynchronize());
    checkCUDA(cudaEventRecord(startEvent));
    for (int i = 0; i < opBaseList.size(); i++)
      opBaseList[i]->forward();
    checkCUDA(cudaEventRecord(endEvent));
    checkCUDA(cudaEventSynchronize(endEvent));
    cudaEventElapsedTime(&milliseconds, startEvent, endEvent);
    total_time += milliseconds;
  }
  return total_time / num_runs;
}

void* Model::allocate_memory(size_t size, const DATATYPE* data_initial)
{
  void* ptr;
  checkCUDA(cudaMalloc(&ptr, size));
  if (data_initial != NULL) {
    checkCUDA(cudaMemcpy(ptr, data_initial, size, cudaMemcpyDefault));
  }
  return ptr;
}

bool Model::copy_memory(DATATYPE* dst, const DATATYPE* src, size_t size)
{
  checkCUDA(cudaMemcpy(dst, src, size, cudaMemcpyDefault));
  return true;
}
