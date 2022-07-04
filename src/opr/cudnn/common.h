#ifndef __COMMON_H_
#define __COMMON_H_

#pragma once

#include <cudnn.h>
#include <memory>

#define CUDNN_CHECK(x) cudnn_check(x, __FILE__, __LINE__)

void cudnn_check(cudnnStatus_t s, std::string file, size_t line);


class HandleManager {
public:
    static std::shared_ptr<HandleManager> get() {
        static std::shared_ptr<HandleManager> manager;
        if(!manager) {
            manager = std::make_shared<HandleManager>();
        }
        return manager;
    }

    cudnnHandle_t* cudnn_handle() {
        return &m_cudnn_handler;
    }

    HandleManager() {
        CUDNN_CHECK(cudnnCreate(&m_cudnn_handler));
    }

    ~HandleManager() {
        CUDNN_CHECK(cudnnDestroy(m_cudnn_handler));
    }

private:
    cudnnHandle_t m_cudnn_handler;
};

#endif
