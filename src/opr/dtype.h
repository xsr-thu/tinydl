#pragma once

#ifndef _DTYPE_H_
#define _DTYPE_H_

using uint64_t = unsigned long;

// #include "tensor.h"

enum class DataType {
    Float32, UInt64, Int64, Bool
};



struct Float32 {
    using T = float;
    static __device__ __forceinline__ float one() {
        return 1.f;
    }
    static __device__ __forceinline__ float zero() {
        return 0.f;    
    }
    static __device__ __forceinline__ float exp(float x) {
        return expf(x);
    }
    static __device__ __forceinline__ float log(float x) {
        return logf(x);
    }
};

struct UInt64 {
    using T = uint64_t;
    static __device__ __forceinline__ uint64_t one() {
        return 1UL;
    }
    static __device__ __forceinline__ uint64_t zero() {
        return 0UL;
    }
};


struct Int64 {
    using T = uint64_t;
    static __device__ __forceinline__ int64_t one() {
        return 1UL;
    }
    static __device__ __forceinline__ int64_t zero() {
        return 0UL;
    }
};



struct Bool {
    using T = bool;
    static __device__ __forceinline__ bool one() {
        return true;
    }
    static __device__ __forceinline__ bool zero() {
        return false;
    }
};


template<typename T>
DataType typeclass_to_enum();

#endif
