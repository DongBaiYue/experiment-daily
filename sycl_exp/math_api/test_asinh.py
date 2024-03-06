
import pybind11_sycl_math
import pybind11_hip_math
import numpy as np

array_size = 1024

A_data = np.random.random([array_size]).astype("float32")
np_output = np.arcsinh(A_data)

hip_output = np.zeros([array_size]).astype("float32")
pybind11_hip_math.asinh(A_data, hip_output)

sycl_output = np.zeros([array_size]).astype("float32")
pybind11_sycl_math.asinh(A_data, sycl_output)

print(np_output)
print(hip_output)
print(sycl_output)
print(np.allclose(np_output, hip_output, rtol=1e-05, atol=1e-05))
print(np.allclose(np_output, sycl_output, rtol=1e-05, atol=1e-05))

for i in range(len(np_output)):
    if abs(np_output[i]-sycl_output[i]) > 1e-3:
        print("%sth, numpy:%s, sycl:%s, hip:%s" % (i, np_output[i], sycl_output[i], hip_output[i]))
