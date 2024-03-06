#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <sycl/sycl.hpp>
namespace py = pybind11;

void my_asinh(py::array input, py::array output) {
    float *input_host = static_cast<float *>(input.request().ptr);
    float *output_host = static_cast<float *>(output.request().ptr);
    int array_size = input.request().shape[0];
    sycl::queue my_gpu_queue( sycl::gpu_selector{} );
    // memory alloc and copy
    float *input_device = static_cast<float* >(sycl::malloc_device(array_size*sizeof(float), my_gpu_queue));
    float *output_device = static_cast<float* >(sycl::malloc_device(array_size*sizeof(float), my_gpu_queue));
    my_gpu_queue.memcpy(input_device, input_host, array_size*sizeof(float));
    my_gpu_queue.memcpy(output_device, output_host, array_size*sizeof(float));
    // kernel
    sycl::range<3> dimGrid(1, 1, 1);
    sycl::range<3> dimBlock(1, 1, array_size);
    my_gpu_queue.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), [=](sycl::nd_item<3> item)
        {
            if ((item.get_local_id(2) < array_size)) {
                output_device[item.get_local_id(2)] = sycl::asinh(input_device[item.get_local_id(2)]);
            };
        });
    });
    my_gpu_queue.wait_and_throw();
    my_gpu_queue.memcpy(input_host, input_device, array_size*sizeof(float));
    my_gpu_queue.memcpy(output_host, output_device, array_size*sizeof(float));
}

void my_atanh(py::array input, py::array output) {
    float *input_host = static_cast<float *>(input.request().ptr);
    float *output_host = static_cast<float *>(output.request().ptr);
    int array_size = input.request().shape[0];
    sycl::queue my_gpu_queue( sycl::gpu_selector{} );
    // memory alloc and copy
    float *input_device = static_cast<float* >(sycl::malloc_device(array_size*sizeof(float), my_gpu_queue));
    float *output_device = static_cast<float* >(sycl::malloc_device(array_size*sizeof(float), my_gpu_queue));
    my_gpu_queue.memcpy(input_device, input_host, array_size*sizeof(float));
    my_gpu_queue.memcpy(output_device, output_host, array_size*sizeof(float));
    // kernel
    sycl::range<3> dimGrid(1, 1, 1);
    sycl::range<3> dimBlock(1, 1, array_size);
    my_gpu_queue.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), [=](sycl::nd_item<3> item)
        {
            if ((item.get_local_id(2) < array_size)) {
                output_device[item.get_local_id(2)] = sycl::atanh(input_device[item.get_local_id(2)]);
            };
        });
    });
    my_gpu_queue.wait_and_throw();
    my_gpu_queue.memcpy(input_host, input_device, array_size*sizeof(float));
    my_gpu_queue.memcpy(output_host, output_device, array_size*sizeof(float));
}

// pybind11_sycl_math 这里约定要与文件名相同
PYBIND11_MODULE(pybind11_sycl_math, m) {
    m.doc() = "pybind11 sycl math plugin"; // optional module docstring
    m.def("asinh", &my_asinh, "sycl asinh function",
          py::arg("input"),
          py::arg("output"));
    m.def("atanh", &my_atanh, "sycl atanh function",
          py::arg("input"),
          py::arg("output"));
}

// clang++ -fsycl -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx906 -O3 -Wall -shared -std=c++17 -fPIC $(python3 -m pybind11 --includes) sycl_math_kernel.cpp -o pybind11_sycl_math$(python3-config --extension-suffix)