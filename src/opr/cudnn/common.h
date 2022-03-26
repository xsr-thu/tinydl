#ifndef __COMMON_H_
#define __COMMON_H_

#pragma once

#include <cudnn.h>

#define CUDNN_CHECK(x) cudnn_check(x)

void cudnn_check(cudnnStatus_t s);

#endif
