#include <sycl/sycl.hpp>
#include <dlfcn.h>

void dlexe(std::string shared_lib_path, std::string func_name){
    //void * so_handler = dlmopen(LM_ID_NEWLM, shared_lib_path.c_str(), RTLD_NOW);
    void * so_handler = dlopen(shared_lib_path.c_str(), RTLD_NOW);
    void (*kernel_func)() = (void (*)())dlsym(so_handler, func_name.c_str());
    kernel_func();
    //dlclose(so_handler);
}

int main(){
    dlexe("sycl_1.so", "func");
    dlexe("sycl_2.so", "func");
    return 0;
}
//clang++ -std=c++17 -O3 -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx906 main.cpp -ldl