#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <hip/hip_runtime.h>
namespace py = pybind11;

__launch_bounds__(1024)
__global__ void _my_asinh(float *in, float * out)
{
	int num = hipThreadIdx_x + hipBlockDim_x * hipBlockIdx_x;
	out[num] = asinh(in[num]);
}

__launch_bounds__(1024)
__global__ void _my_atanh(float *in, float * out)
{
	int num = hipThreadIdx_x + hipBlockDim_x * hipBlockIdx_x;
	out[num] = atanh(in[num]);
}

void my_asinh(py::array input, py::array output) {
    float *input_host = static_cast<float *>(input.request().ptr);
    float *output_host = static_cast<float *>(output.request().ptr);
    int array_size = input.request().shape[0];
    float *input_device, *output_device;
    hipMalloc((void**)&input_device, array_size*sizeof(float));
    hipMalloc((void**)&output_device, array_size*sizeof(float));
    hipMemcpy((void*)input_device, (void*)input_host, array_size*sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy((void*)output_device, (void*)output_host, array_size*sizeof(float), hipMemcpyHostToDevice);
    int dimGrid = 1;
    int dimBlock = array_size;
    _my_asinh<<<dimGrid, dimBlock>>>(input_device, output_device);
    hipDeviceSynchronize();
    hipMemcpy((void*)input_host, (void*)input_device, array_size*sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy((void*)output_host, (void*)output_device, array_size*sizeof(float), hipMemcpyDeviceToHost);
}

void my_atanh(py::array input, py::array output) {
    float *input_host = static_cast<float *>(input.request().ptr);
    float *output_host = static_cast<float *>(output.request().ptr);
    int array_size = input.request().shape[0];
    float *input_device, *output_device;
    hipMalloc((void**)&input_device, array_size*sizeof(float));
    hipMalloc((void**)&output_device, array_size*sizeof(float));
    hipMemcpy((void*)input_device, (void*)input_host, array_size*sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy((void*)output_device, (void*)output_host, array_size*sizeof(float), hipMemcpyHostToDevice);
    int dimGrid = 1;
    int dimBlock = array_size;
    _my_atanh<<<dimGrid, dimBlock>>>(input_device, output_device);
    hipDeviceSynchronize();
    hipMemcpy((void*)input_host, (void*)input_device, array_size*sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy((void*)output_host, (void*)output_device, array_size*sizeof(float), hipMemcpyDeviceToHost);
}

// pybind11_hip_math 这里约定要与文件名相同
PYBIND11_MODULE(pybind11_hip_math, m) {
    m.doc() = "pybind11 hip math plugin"; // optional module docstring
    m.def("asinh", &my_asinh, "hip asinh function",
          py::arg("input"),
          py::arg("output"));
    m.def("atanh", &my_atanh, "hip atanh function",
          py::arg("input"),
          py::arg("output"));
}

// hipcc -O3 -Wall -shared -std=c++17 -fPIC $(python3 -m pybind11 --includes) hip_math_kernel.cpp -o pybind11_hip_math$(python3-config --extension-suffix)