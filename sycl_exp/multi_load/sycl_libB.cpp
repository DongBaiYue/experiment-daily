#include <CL/sycl.hpp>
using namespace sycl;

#ifdef __cplusplus
extern "C"
#endif
void func() {
    sycl::property_list q_prop{
        sycl::property::queue::in_order()};
    queue* q= new queue(q_prop);
    std::cout << "B" << std::endl;
    q->submit([&](sycl::handler &h) {
        sycl::stream os(1024, 768, h);
        h.parallel_for<class lib>(32, [=](sycl::id<1> i) {
            os<<"B";
        });
    }).wait();
    delete q;
}
//clang++ -std=c++17 -O3 -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx906 -fPIC -shared sycl_libB.cpp -o sycl_libB.so