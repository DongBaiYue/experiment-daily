#include <CL/sycl.hpp>
using namespace sycl;
class lib;

#ifdef __cplusplus
extern "C"
#endif
void func() {
    sycl::property_list q_prop{
        sycl::property::queue::in_order()};
    queue* q= new queue(q_prop);
    std::cout << "A" << std::endl;
    auto myDevice = q->get_device();
    q->submit([&](sycl::handler &h) {
        sycl::stream os(1024, 768, h);
        h.parallel_for<lib>(16, [=](sycl::id<1> i) {
            os<<"A";
        });
    }).wait();
    delete q;
}

#ifdef __cplusplus
extern "C"
#endif
void func2() {
    sycl::property_list q_prop{
        sycl::property::queue::in_order()};
    queue* q= new queue(q_prop);
    std::cout << "A" << std::endl;
    auto myDevice = q->get_device();
    q->submit([&](sycl::handler &h) {
        sycl::stream os(1024, 768, h);
        h.parallel_for<lib>(16, [=](sycl::id<1> i) {
            os<<"A";
        });
    }).wait();
    delete q;
}
//clang++ -std=c++17 -O3 -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx906 -fPIC -shared sycl_libA.cpp -o sycl_libA.so