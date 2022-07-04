#include "dtype.h"

template<>
DataType typeclass_to_enum<Float32>() {
    return DataType::Float32;
}

template<>
DataType typeclass_to_enum<UInt64>() {
    return DataType::UInt64;
}

template<>
DataType typeclass_to_enum<Int64>() {
    return DataType::Int64;
}

template<>
DataType typeclass_to_enum<Bool>() {
    return DataType::Bool;
}

template<>
DataType typeclass_to_enum<float>() {
    return DataType::Float32;
}

template<>
DataType typeclass_to_enum<uint64_t>() {
    return DataType::UInt64;
}

template<>
DataType typeclass_to_enum<int64_t>() {
    return DataType::Int64;
}

template<>
DataType typeclass_to_enum<bool>() {
    return DataType::Bool;
}

