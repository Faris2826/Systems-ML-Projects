//==============================================================================
//
//  @@-COPYRIGHT-START-@@
//
//  Copyright (c) 2023, Qualcomm Innovation Center, Inc. All rights reserved.
//
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions are met:
//
//  1. Redistributions of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//  2. Redistributions in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//  3. Neither the name of the copyright holder nor the names of its contributors
//     may be used to endorse or promote products derived from this software
//     without specific prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
//  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
//  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
//  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
//  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
//  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
//  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
//  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//  POSSIBILITY OF SUCH DAMAGE.
//
//  SPDX-License-Identifier: BSD-3-Clause
//
//  @@-COPYRIGHT-END-@@
//
//==============================================================================

#include "QcQuantizeOp.h"
#include "AimetOpUtils.h"
#include "trim_functions.hpp"

#include <vector>
#include <iostream>
#include <type_traits>


#ifdef ONNX_CUDA
static DlQuantization::CudaAllocator cudaAllocator;
#endif
static DlQuantization::CpuAllocator cpuAllocator;

template<typename T>
QcQuantizeOp<T>::QcQuantizeOp(const OrtApi* api, const OrtKernelInfo* info) : api_(*api), info_(info)
{
    int64_t quantInfoPointer;
    api->KernelInfoGetAttribute_int64(info_, "quant_info", &quantInfoPointer);
    quantInfo = reinterpret_cast<struct QcQuantizeInfo*>(quantInfoPointer);
}

template<typename T>
void ConvertFloat16ToFloat32(const T* inputData, float*& convertedPtr, size_t size, bool useCuda, void* stream)
{
    if (useCuda)
    {
        #ifdef ONNX_CUDA
            DlQuantization::convertFp16ToFloatKernelForGPU(reinterpret_cast<const __half*>(inputData), size, convertedPtr, stream);
        #else
            throw std::runtime_error("Cannot call CUDA kernel for type conversion. Not compiled for GPU mode.");
        #endif
    }
    else
    {
        for (size_t i = 0; i < size; ++i)
        {
            convertedPtr[i] = inputData[i].ToFloat();
        }
    }
}

template<typename T>
void ConvertFloat32ToFloat16(float* resultPtr, T* convertedResult, size_t size, bool useCuda, void* stream)
{
    if (useCuda)
    {
        #ifdef ONNX_CUDA
            DlQuantization::convertFloatToFp16KernelForGPU(resultPtr, size, reinterpret_cast<__half*>(convertedResult), stream);
        #else
            throw std::runtime_error("Cannot call CUDA kernel for type conversion. Not compiled for GPU mode.");
        #endif
    }
    else
    {
        for (size_t i = 0; i < size; ++i)
        {
            convertedResult[i] = Ort::Float16_t(resultPtr[i]);
        }
    }
}

template<typename T>
void QcQuantizeOp<T>::computeImpl(const Ort::Custom::Tensor<T>& input, Ort::Custom::Tensor<T>& output,
                               void* stream, bool useCuda, DlQuantization::IAllocator* allocator)
{
    // Setup inputs
    auto inputShape = input.Shape();
    auto size     = input.NumberOfElement();
    auto result     = output.Allocate(inputShape);

    DlQuantization::TensorQuantizerOpMode opMode = quantInfo->opMode;
    // Disable unused quantizers
    if (!quantInfo->enabled)
    {
        opMode = DlQuantization::TensorQuantizerOpMode::passThrough;
    }

    const float* inputPtr = nullptr;
    float* resultPtr = nullptr;

    if constexpr(std::is_same<T, Ort::Float16_t>::value)
    {
        // FLOAT16 Implementation
        const T* inputData = input.Data();
        float* inputPtrIntermediate = static_cast<float*>(allocator->allocateRaw(size * sizeof(float)));;

        ConvertFloat16ToFloat32(inputData, inputPtrIntermediate, size, useCuda, stream);

        inputPtr = inputPtrIntermediate;
        resultPtr = static_cast<float*>(allocator->allocateRaw(size * sizeof(float)));

    }
    else
    {
        // FLOAT32 Implementation
        inputPtr = input.Data();
        resultPtr = result;
    }

    if (quantInfo->isIntDataType)
    {
        modeSpecificActionBroadcastInt(inputPtr, resultPtr, inputShape, quantInfo->tensorQuantizer.get(), opMode,
            quantInfo->useSymmetricEncoding, allocator, useCuda, stream);
    }
    else
    {
        modeSpecificActionFloat(inputPtr, size, resultPtr, opMode, allocator, useCuda, stream);
    }

    if constexpr(std::is_same<T, Ort::Float16_t>::value)
    {

        // FLOAT16 Implementation
        ConvertFloat32ToFloat16(resultPtr, result, size, useCuda, stream);

        if (useCuda)
        {
            #ifdef ONNX_CUDA
                cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
            #endif
        }

        allocator->deleteRaw((void*)inputPtr);
        allocator->deleteRaw((void*)resultPtr);

    }

    // We only ever need to run in oneShotQuantizeDequantize once, afterwards just use quantizeDequantize
    if (opMode == DlQuantization::TensorQuantizerOpMode::oneShotQuantizeDequantize)
    {
        quantInfo->opMode = DlQuantization::TensorQuantizerOpMode::quantizeDequantize;
    }
}

template<typename T>
struct QcQuantizeOpCpu : QcQuantizeOp<T>
{
    using QcQuantizeOp<T>::QcQuantizeOp;

    void Compute(const Ort::Custom::Tensor<T>& input, Ort::Custom::Tensor<T>& output)
    {
        this->computeImpl(input, output, nullptr, false, &cpuAllocator);
    }
};

template<typename T>
struct QcQuantizeOpCpuFloat16 : QcQuantizeOp<T>
{
    using QcQuantizeOp<T>::QcQuantizeOp;

    void Compute(const Ort::Custom::Tensor<T>& input, Ort::Custom::Tensor<T>& output)
    {
        this->computeImpl(input, output, nullptr, false, &cpuAllocator);
    }
};



#ifdef ONNX_CUDA

template<typename T>
struct QcQuantizeOpCuda : QcQuantizeOp<T>
{
    using QcQuantizeOp<T>::QcQuantizeOp;

    void Compute(const Ort::Custom::CudaContext& cuda_ctx, const Ort::Custom::Tensor<T>& input,
                 Ort::Custom::Tensor<T>& output)
    {
        cudaStream_t stream = cuda_ctx.cuda_stream;
        this->computeImpl(input, output, stream, true, &cudaAllocator);
    }
};

template<typename T>
struct QcQuantizeOpCudaFloat16 : QcQuantizeOp<T>
{
    using QcQuantizeOp<T>::QcQuantizeOp;

    void Compute(const Ort::Custom::CudaContext& cuda_ctx, const Ort::Custom::Tensor<T>& input,
                 Ort::Custom::Tensor<T>& output)
    {
        cudaStream_t stream = cuda_ctx.cuda_stream;
        this->computeImpl(input, output, stream, true, &cudaAllocator);
    }
};

#endif


void RegisterOps(Ort::CustomOpDomain& domain)
{
    static const std::unique_ptr<Ort::Custom::OrtLiteCustomOp> qcQuantCpuOpPointer {
        Ort::Custom::CreateLiteCustomOp<QcQuantizeOpCpu<float>>("QcQuantizeOp", "CPUExecutionProvider")};
    domain.Add(qcQuantCpuOpPointer.get());

    static const std::unique_ptr<Ort::Custom::OrtLiteCustomOp> qcQuantCpuOpFloat16Pointer {
        Ort::Custom::CreateLiteCustomOp<QcQuantizeOpCpuFloat16<Ort::Float16_t>>("QcQuantizeOp", "CPUExecutionProvider")};
    domain.Add(qcQuantCpuOpFloat16Pointer.get());
#ifdef ONNX_CUDA
    static const std::unique_ptr<Ort::Custom::OrtLiteCustomOp> qcQuantCudaOpPointer {
        Ort::Custom::CreateLiteCustomOp<QcQuantizeOpCuda<float>>("QcQuantizeOp", "CUDAExecutionProvider")};
    domain.Add(qcQuantCudaOpPointer.get());

    static const std::unique_ptr<Ort::Custom::OrtLiteCustomOp> qcQuantCudaOpFloat16Pointer {
        Ort::Custom::CreateLiteCustomOp<QcQuantizeOpCudaFloat16<Ort::Float16_t>>("QcQuantizeOp", "CUDAExecutionProvider")};
    domain.Add(qcQuantCudaOpFloat16Pointer.get());
#endif
}
