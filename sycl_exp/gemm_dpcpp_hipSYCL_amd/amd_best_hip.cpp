//hipcc -O3 --offload-arch=gfx906 amd_best_hip.cpp -o amd_hip

#include "hip/hip_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>

#include <cmath>

#define FLOAT float
#define INT int
#define CEIL_DIV(m,n) ( (m) + (n) - 1 ) / (n)

void randomize_matrix(FLOAT* mat, int N);
void copy_matrix(FLOAT *src, FLOAT *dest, int n);
bool verify_matrix(FLOAT *mat1, FLOAT *mat2, int n);

void test_mysgemm_v9(INT M, INT N, INT K, FLOAT alpha, FLOAT *A, FLOAT *B, FLOAT beta, FLOAT *C);

#define A(i,j) A[(i) + (j)*lda]
#define B(i,j) B[(i) + (j)*ldb]
#define C(i,j) C[(i) + (j)*ldc]
// TEMPLATE:param17: 7(Mb=128,7bit)
// #define sa9(i,j) sa9[((j)<<7) + (i)]
#define sa9(i,j) sa9[((j)<<param17) + (i)]
// TEMPLATE:param18: 7(Nb=128,7bit)
// #define sb9(i,j) sb9[((j)<<7) + (i)]
#define sb9(i,j) sb9[((j)<<param18) + (i)]
// #define MS_9 128
// #define NS_9 128
// #define KS_9 8
//v1 += v2 * s3, vector scaling
#define vscal(v1, v2, s3)\
    v1.x+=v2.x*s3;\
    v1.y+=v2.y*s3;\
    v1.z+=v2.z*s3;\
    v1.w+=v2.w*s3;
//v1 = alpha * v2 + beta * v3, simd fma
#define simd_axpby(v1, alpha, v2, beta, v3)\
    v1.x=alpha*v2.x+beta*v3.x;\
    v1.y=alpha*v2.y+beta*v3.y;\
    v1.z=alpha*v2.z+beta*v3.z;\
    v1.w=alpha*v2.w+beta*v3.w;
#define vload(v1,addr)\
    v1 = *((float4 *)(addr));
#define vstore(addr,v1)\
    *((float4 *)(addr)) = v1;

void test_mysgemm_v1(INT M, INT N, INT K, FLOAT alpha, FLOAT *A, FLOAT *B, FLOAT beta, FLOAT *C);

__global__ __launch_bounds__(1024)
void mysgemm_v1(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C){
    int lda = M, ldb = K, ldc = M;
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;
    // A,B 现在指向了对应的行列的地址；
    // 首先划定线程块对应的位置
    A = &A((bx<<5),0);
    B = &B(0,(by<<5));
    C = &C((bx<<5),(by<<5));
    float tmp=0.;
    for (int k_count = 0; k_count<K; k_count++){
        // 在线程块中做一个累加
        tmp += A(tx, k_count) * B(k_count, ty);
    }
    C(tx,ty) = alpha * tmp + beta*C(tx,ty);
}

// cache blocking version, without register-level data re-use
// with memory coelascing on shared memory
// more workloads per thread. 8x8 micro kernel.
// adopt vetorized load/store
// TEMPLATE:
template <
    const int Mb,   // Mb = 128,7bit "1" 元素, 一个block占据的元素数量
    const int Kb,   // Kb = 8,  3bit
    const int Nb,   // Nb = 128,7bit
    const int Mw,   // Mw = 32, 5bit "1" 元素, 一个 warp 占据的元素数量
    const int Nw,   // Nw = 64, 6bit        
    const int Mt,   // Mt = 8,  3bit        , 一个 thread 占据的元素数量
    const int Kt, 
    const int Nt,   // Nt = 8        "1" 元素
    const int param0,    // param0 = set1(bitNum((Mb/Mw)))
    const int param1,    // param1 = bitNum((Mb/Mw))
    const int param2,    // param2 = set1(bitNum((Mw/Mt)))
    const int param3,    // param3 = set1(bitNum((Mw/Mt))) 
    const int threadNums,// param4 = (Mb/Mt)*(Nb/Nt)
    const int thNBit,    // param5 = bitNum((Mb/Mt)*(Nb/Nt))
    const int param6,    // param6 = param5 - bitNum(Nb)
    const int param7,    // param7 = set1(param6)
    const int param8,    // param8 = bitNum(Kb) - param7
    const int param9,    // param9 = bitNum(Kb)
    const int param10,   // param10 = bitNum(Mw) 
    const int param11,   // param11 = bitNum(Mt) 
    const int param12,   // param12 = bitNum(Nw)
    const int param13,   // param13 = bitNum(Nt)
    const int param14, 	 // param14 = param5 - bitNum(Kb)
	const int param15,   // param15 = set1(param14)
	const int param16,   // param16 = bitNum(Mb) - param14
	const int param17,   // param17 = bitNum(Mb)
	const int param18,   // param18 = bitNum(Nb)
	const int param19,   // param19 = Mb*Kb
	const int param20, 	 // param20 = Nb*Kb
	const int param21, 	 // param21 = Mt/4
	const int param22,   // param22 = Nt/4
	const int param23,   // param23 = Mt*Nt/4
    const int param24,   // param24 = bitNum((Mw/Mt))   
    const int param25    // param25 = bitNum(param19/4)-thNBit
    >
__global__ __launch_bounds__(512)
void mysgemm_v9(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C){

    int lda = M, ldb = K, ldc = M;
    int tx = threadIdx.x;
    int bx = blockIdx.x, by = blockIdx.y;
    int warp_id = tx>>5;    
    int lane_id = tx&31;    
    // TEMPLATE:param{0,1}: 3(Mb=128,Mw=32,128/32=4,2bit,0b11=3), 2(Mb=128,Mw=32,128/32=4,2bit)
    // int warp_row = warp_id & 3, warp_col = warp_id >> 2;    // 从而得知256个线程里的8个warp是怎么排的：4行2列；
    int warp_row = warp_id & param0, warp_col = warp_id >> param1;    // 从而得知256个线程里的8个warp是怎么排的：4行2列；

    // TEMPLATE:param{3,24}: 3(Mw=32,Mt=8,32/8=4,2bit,ob11=3), 2(Mw=32,Mt=8,32/8=4,2bit) 
    // int row_w = lane_id&3, col_w = lane_id>>2;             // 从而得知一个 warp 中的32个线程是怎么排布的：4行8列；
    int row_w = lane_id&param3, col_w = lane_id>>param24;             // 从而得知一个 warp 中的32个线程是怎么排布的：4行8列；

    // TEMPLATE:param6: col_b 1 取高7位(Nb=128,7bit,共thNBit=8bit,8bit-7bit=1)   
    // TEMPLATE:param{7,8}: row_b 1 (8bit还剩1bit) 2(Kb=8,3bit,3bit-1bit=2bit)
    // int row_b = (tx&1)<<2, col_b = tx>>1;               // 从而得知矩阵b中一个元素的位置(in a block)；3位行号(2位扩展)，7位列号，对应1024个元素
    int row_b = (tx&param7)<<param8, col_b = tx>>param6;               // 从而得知矩阵b中一个元素的位置(in a block)；3位行号(2位扩展)，7位列号，对应1024个元素

    // TEMPLATE:param9: 3(Kb=8,3bit)
    // int lda8 = lda<<3;  
    int lda8 = lda<<param9;  

    // TEMPLATE:param10: 5(Mw=32,5bit) 
    // TEMPLATE:param11: 3(Mt=8,3bit) 
    // TEMPLATE:param12: 6(Nw=64,6bit) 
    // TEMPLATE:param13: 3(Nt=8,3bit)
    // int row_c = (warp_row<<5) + (row_w<<3), col_c = (warp_col<<6) + (col_w<<3); // 从而得知结果矩阵中一个元素的位置(in a block)；row_c, col_c
    int row_c = (warp_row<<param10) + (row_w<<param11), col_c = (warp_col<<param12) + (col_w<<param13); // 从而得知结果矩阵中一个元素的位置(in a block)；row_c, col_c

    // TEMPLATE:param14: col_a 5(thNBit=8bit,Kb=8,3bit, 8bit-3bit=5)
    // TEMPLATE:param{15,16}: 31(col_a 5bit --> 31) 2(Mb=128,7bit,7bit-5bit=2)
    // int row_a = (tx&31)<<2, col_a = tx>>5;                                      // 从而得知矩阵a中一个元素的位置(in a block);
    int row_a = (tx&param15)<<param16, col_a = tx>>param14;                                      // 从而得知矩阵a中一个元素的位置(in a block);

    // TEMPLATE:param17: 7(Mb=128,7bit)
    // A = &A((bx<<7),0);  // 指向A的第一个元素(in all blocks)
    A = &A((bx<<param17),0);  // 指向A的第一个元素(in all blocks)

    // TEMPLATE:param18: 7(Nb=128,7bit)
    // B = &B(0,(by<<7));  // 指向B的第一个元素(in all blocks)
    B = &B(0,(by<<param18));  // 指向B的第一个元素(in all blocks)

    // TEMPLATE:param{17,18}: 7(Mb=128,7bit) 7(Nb=128,7bit)
    // C = &C((bx<<7),(by<<7));//the TB size is 128.   // 指向B的第一个元素(in a block)
    C = &C((bx<<param17),(by<<param18));//the TB size is 128.   // 指向B的第一个元素(in a block)

    // TEMPLATE:param19: 1024(Mb=128,Kb=8,128*8=1024)
    __shared__ float sa9[2048];

    // TEMPLATE:param20: 1024(Nb=128,Kb=8,128*8=1024)
    __shared__ float sb9[2048];

    // TEMPLATE:param21: 2(Mt=8,8/4=2)
    // TEMPLATE:param22: 2(Nt=8,8/4=2)
    // TEMPLATE:param23: 16(Mt=8,Nt=8,8*8/4=16)
    // float4 Av1, Av2, Bv1, Bv2, Cv[16], Cres[16];
    float4 Av[param21], Bv[param22], Cv[param23], Cres[param23];

    memset(Cres, 0, sizeof(Cres));//clear registers
    // TEMPLATE:Kb: KS_9=128(Kb=128)
    // for (int k_count = 0; k_count<K; k_count+=KS_9){
    for (int k_count = 0; k_count<K; k_count+=Kb){
        /*packing A and B into shared memory*/
        // 每个线程都要负责将需要的元素搬入共享内存，但是这里弄得很巧合——正好是一个 vector 的大小

        // TEMPLATE:这一段代码主要使用来填充 shared memory
        // vload(Av1, &A(row_a,col_a))
        // vload(Bv1, &B(row_b,col_b))
        // ((float4 *)sa9)[tx] = Av1;
        // sb9(col_b,row_b)=Bv1.x;
        // sb9(col_b,row_b+1)=Bv1.y;
        // sb9(col_b,row_b+2)=Bv1.z;
        // sb9(col_b,row_b+3)=Bv1.w;
        // A+=lda8;B+=8;
        // shared memory 是竖着放的

        // int limit= Mb*Kb/4/threadNums;
        int offsetA = 0;
        int offsetB = 0;
        #pragma unroll
        for (int cnt = 0; cnt < Mb*Kb/4/threadNums; cnt++) {
            vload(Av[0], &A(row_a + offsetA, col_a))
                // TEMPLATE: param25: (param19/4 --> sa 有多少个 vector;
                // bitNum(param19/4) --> 需要多少位; `bitNum(param19/4)-thNBit`
                // --> 需要扩多少bit
                ((float4 *)sa9)[(tx << param25) + cnt] = Av[0];
            offsetA += 4;
        }
        #pragma unroll
        for (int cnt = 0; cnt < Nb*Kb/4/threadNums; cnt++) {
            vload(Bv[0], &B(row_b + offsetB, col_b))
                // TEMPLATE: param25: (param19/4 --> sa 有多少个 vector;
                // bitNum(param19/4) --> 需要多少位; `bitNum(param19/4)-thNBit`
                // --> 需要扩多少bit
                sb9(col_b, row_b + offsetB) = Bv[0].x;
            sb9(col_b, row_b + 1 + offsetB) = Bv[0].y;
            sb9(col_b, row_b + 2 + offsetB) = Bv[0].z;
            sb9(col_b, row_b + 3 + offsetB) = Bv[0].w;
            offsetB += 4;
        }
        A+=lda8;B+=Kb;
        /*
        DPCT1065:0: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        __syncthreads();

#pragma unroll
        // TEMPLATE: KS_9=8(Kb=8)
        // for (int inner_k_count=0;inner_k_count<KS_9;inner_k_count++){ 
        for (int inner_k_count=0;inner_k_count<Kb;inner_k_count++){ 
            // 计算乘法，但是这里做了手动的展开；// 模板需要修改

            // TEMPLATE: 把 load SA 和 SB 的vector
            #pragma unroll
            for (int cnt=0; cnt < param21; cnt++) {            // 两行 vector
                vload(Av[cnt], &sa9(row_c+cnt*4,inner_k_count))
            }
            #pragma unroll
            // for (int cnt=0; cnt < Mt/2; cnt++) {            // 两行 vector
            for (int cnt=0; cnt < param22; cnt++) {            // 两行 vector
                vload(Bv[cnt], &sb9(col_c+cnt*4,inner_k_count))
            }

            #pragma unroll
            for (int cnt_i = 0; cnt_i < param22; cnt_i++) { // 0, 1, Bv[i]
            // for (int cnt_i = 0; cnt_i < 4; cnt_i++) { // 0, 1, Bv[i]
                #pragma unroll
                for (int cnt_j = 0; cnt_j < param21; cnt_j++) { // 0, 1, Bv[i]
                    // vscal(Cres[cnt_i*8+0*2+cnt_j], Av[cnt_j], Bv[cnt_i].x)
                    vscal(Cres[cnt_i * 4 * param21 + 0 * param21 + cnt_j],
                          Av[cnt_j], Bv[cnt_i].x)
                }
                #pragma unroll
                for (int cnt_j = 0; cnt_j < param21; cnt_j++) { // 0, 1, Bv[i]
                    vscal(Cres[cnt_i * 4 * param21 + 1 * param21 + cnt_j],
                          Av[cnt_j], Bv[cnt_i].y)
                }
                #pragma unroll
                for (int cnt_j = 0; cnt_j < param21; cnt_j++) { // 0, 1, Bv[i]
                    vscal(Cres[cnt_i * 4 * param21 + 2 * param21 + cnt_j],
                          Av[cnt_j], Bv[cnt_i].z)
                }
                #pragma unroll
                for (int cnt_j = 0; cnt_j < param21; cnt_j++) { // 0, 1, Bv[i]
                    vscal(Cres[cnt_i * 4 * param21 + 3 * param21 + cnt_j],
                          Av[cnt_j], Bv[cnt_i].w)
                }

            }
            
            // vload(Av1, &sa9(row_c,inner_k_count))
            // vload(Av2, &sa9(row_c+4,inner_k_count))
            // vload(Bv1, &sb9(col_c,inner_k_count))
            // vload(Bv2, &sb9(col_c+4,inner_k_count))
            // vscal(Cres[0], Av1, Bv1.x)
            // vscal(Cres[1], Av2, Bv1.x)
            // vscal(Cres[2], Av1, Bv1.y)
            // vscal(Cres[3], Av2, Bv1.y)
            // vscal(Cres[4], Av1, Bv1.z)
            // vscal(Cres[5], Av2, Bv1.z)
            // vscal(Cres[6], Av1, Bv1.w)
            // vscal(Cres[7], Av2, Bv1.w)
            // vscal(Cres[8], Av1, Bv2.x)
            // vscal(Cres[9], Av2, Bv2.x)
            // vscal(Cres[10], Av1, Bv2.y)
            // vscal(Cres[11], Av2, Bv2.y)
            // vscal(Cres[12], Av1, Bv2.z)
            // vscal(Cres[13], Av2, Bv2.z)
            // vscal(Cres[14], Av1, Bv2.w)
            // vscal(Cres[15], Av2, Bv2.w)
        }
        /*
        DPCT1065:1: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
       __syncthreads();
    }
    // 
    // TEMPLATE: 读取 C 的值
    #pragma unroll
    for (int col_index=0; col_index < Nt; col_index++) {
        #pragma unroll
        for (int row_index=0; row_index < Mt/4; row_index++) {
            vload(Cv[col_index*(Mt/4) + row_index], &C(row_c+row_index*4,col_c+col_index))
        }
    }
    // vload(Cv[0], &C(row_c,col_c))
    // vload(Cv[1], &C(row_c+4,col_c))
    // vload(Cv[2], &C(row_c,col_c+1))
    // vload(Cv[3], &C(row_c+4,col_c+1))
    // vload(Cv[4], &C(row_c,col_c+2))
    // vload(Cv[5], &C(row_c+4,col_c+2))
    // vload(Cv[6], &C(row_c,col_c+3))
    // vload(Cv[7], &C(row_c+4,col_c+3))
    // vload(Cv[8], &C(row_c,col_c+4))
    // vload(Cv[9], &C(row_c+4,col_c+4))
    // vload(Cv[10], &C(row_c,col_c+5))
    // vload(Cv[11], &C(row_c+4,col_c+5))
    // vload(Cv[12], &C(row_c,col_c+6))
    // vload(Cv[13], &C(row_c+4,col_c+6))
    // vload(Cv[14], &C(row_c,col_c+7))
    // vload(Cv[15], &C(row_c+4,col_c+7))
    
    // TEMPLATE: a*x + b*y
    #pragma unroll
    for (int i_cnt = 0; i_cnt < Mt*Nt/4; i_cnt++) {
        simd_axpby(Cres[i_cnt],alpha,Cres[i_cnt],beta,Cv[i_cnt])
    }

    // simd_axpby(Cres[0],alpha,Cres[0],beta,Cv[0])
    // simd_axpby(Cres[1],alpha,Cres[1],beta,Cv[1])
    // simd_axpby(Cres[2],alpha,Cres[2],beta,Cv[2])
    // simd_axpby(Cres[3],alpha,Cres[3],beta,Cv[3])

    // simd_axpby(Cres[4],alpha,Cres[4],beta,Cv[4])
    // simd_axpby(Cres[5],alpha,Cres[5],beta,Cv[5])
    // simd_axpby(Cres[6],alpha,Cres[6],beta,Cv[6])
    // simd_axpby(Cres[7],alpha,Cres[7],beta,Cv[7])

    // simd_axpby(Cres[8],alpha,Cres[8],beta,Cv[8])
    // simd_axpby(Cres[9],alpha,Cres[9],beta,Cv[9])
    // simd_axpby(Cres[10],alpha,Cres[10],beta,Cv[10])
    // simd_axpby(Cres[11],alpha,Cres[11],beta,Cv[11])

    // simd_axpby(Cres[12],alpha,Cres[12],beta,Cv[12])
    // simd_axpby(Cres[13],alpha,Cres[13],beta,Cv[13])
    // simd_axpby(Cres[14],alpha,Cres[14],beta,Cv[14])
    // simd_axpby(Cres[15],alpha,Cres[15],beta,Cv[15])

    // TEMPLATE: 存回到C
    #pragma unroll
    for (int col_index=0; col_index < Nt; col_index++) {
        #pragma unroll
        for (int row_index=0; row_index < Mt/4; row_index++) {
            vstore(&C(row_c+row_index*4,col_c+col_index), Cres[col_index*(Mt/4) + row_index])
        }
    }

    // vstore(&C(row_c,col_c), Cres[0])
    // vstore(&C(row_c+4,col_c), Cres[1])
    // vstore(&C(row_c,col_c+1), Cres[2])
    // vstore(&C(row_c+4,col_c+1), Cres[3])
    // vstore(&C(row_c,col_c+2), Cres[4])
    // vstore(&C(row_c+4,col_c+2), Cres[5])
    // vstore(&C(row_c,col_c+3), Cres[6])
    // vstore(&C(row_c+4,col_c+3), Cres[7])
    // vstore(&C(row_c,col_c+4), Cres[8])
    // vstore(&C(row_c+4,col_c+4), Cres[9])
    // vstore(&C(row_c,col_c+5), Cres[10])
    // vstore(&C(row_c+4,col_c+5), Cres[11])
    // vstore(&C(row_c,col_c+6), Cres[12])
    // vstore(&C(row_c+4,col_c+6), Cres[13])
    // vstore(&C(row_c,col_c+7), Cres[14])
    // vstore(&C(row_c+4,col_c+7), Cres[15])
}

int main() {
        // 我们设定24个测试矩阵。大 (i+1)*2^8
    int SIZE[24];
    for (int i=0;i<24;i++) SIZE[i]=(i+1)<<8;
	// int upper_limit=4;
	int upper_limit=(sizeof(SIZE)/sizeof(int));

    // 开辟 host 和 device 需要的内存指针，后续进行内存的分配
    FLOAT *A=NULL,*B=NULL,*C=NULL,*C_ref=NULL;//host matrices
    FLOAT *dA=NULL,*dB=NULL,*dC=NULL,*dC_ref=NULL;//device matrices
    // alpha 和 beta 两个系数，我们手动指定
    FLOAT alpha = 1.0, beta = 0.;//two arbitary input parameters
	
    // 记录一次计算所需要的时间
    // 并且创建好开始和终止 EVENT,用来收集时间
    float elapsed_time;
    //dpct::event_ptr beg, end;
    std::chrono::time_point<std::chrono::steady_clock> beg_ct1;
    std::chrono::time_point<std::chrono::steady_clock> end_ct1;
    //beg = new sycl::event();
    //end = new sycl::event();

        int max_size;
	max_size = SIZE[upper_limit - 1];

    // 进行内存的分配, host, 并且直接按照最大的size进行分配
    A=(FLOAT *)malloc(sizeof(FLOAT)*max_size*max_size);
    B=(FLOAT *)malloc(sizeof(FLOAT)*max_size*max_size);
    C=(FLOAT *)malloc(sizeof(FLOAT)*max_size*max_size);
    C_ref=(FLOAT *)malloc(sizeof(FLOAT)*max_size*max_size);
	
    // 用来host的分配的内存进行初始化
    randomize_matrix(A,max_size*max_size);randomize_matrix(B,max_size*max_size);
    randomize_matrix(C,max_size*max_size);copy_matrix(C,C_ref,max_size*max_size);

    // 分配 device 的内存，然后直接把host的给复制过来
    hipMalloc((void**) &dA, sizeof(FLOAT)*max_size*max_size);
    hipMalloc((void**) &dB, sizeof(FLOAT)*max_size*max_size);
    hipMalloc((void**) &dC, sizeof(FLOAT)*max_size*max_size);
    hipMalloc((void**) &dC_ref, sizeof(FLOAT)*max_size*max_size);
    hipMemcpy(dA, A, sizeof(FLOAT)*max_size*max_size, hipMemcpyHostToDevice);
    hipMemcpy(dB, B, sizeof(FLOAT)*max_size*max_size, hipMemcpyHostToDevice);
    hipMemcpy(dC, C, sizeof(FLOAT)*max_size*max_size, hipMemcpyHostToDevice);
    hipMemcpy(dC_ref, C_ref, sizeof(FLOAT)*max_size*max_size, hipMemcpyHostToDevice);

        // 依次对我们要测的矩阵进行处理
	int m, n, k; // 输入矩阵的 size
	int n_count, N=4;
    for (int i_count=0;i_count<upper_limit;i_count++){
		// DEBUG: 某个 size
		if (i_count < 0 || i_count >= 23+1) continue;
        // 获取矩阵的三个轴大小
        m=n=k=SIZE[i_count];
        printf("\nM=N=K= %d :\n",m);

		// 调用我们的 kernel 情况
		test_mysgemm_v9(m, n, k, alpha, dA, dB, beta, dC);
                hipDeviceSynchronize();

        // 调用 naive 版本进行验证
        test_mysgemm_v1(m, n, k, alpha, dA, dB, beta, dC_ref);
                hipDeviceSynchronize();

                // 把我们的kernel计算的结果复制到 host
                hipMemcpy(C, dC, sizeof(FLOAT)*m*n, hipMemcpyDeviceToHost);
        hipMemcpy(C_ref, dC_ref, sizeof(FLOAT)*m*n, hipMemcpyDeviceToHost);
                hipDeviceSynchronize();

                if (!verify_matrix(C_ref,C,m*n)) {
			printf("Failed to pass the correctness verification!!!!!!!!!!\n");
			exit(-3);
		} else {
            printf("Pass.\n");
        }


        // 然后来计算时间？算10遍，然后我们再去取平均值；
                /*
                DPCT1012:4: Detected kernel execution time measurement pattern
                and generated an initial code for time measurements in SYCL. You
                can change the way time is measured depending on your goals.
                */
                beg_ct1 = std::chrono::steady_clock::now();
                for (n_count=0;n_count<N;n_count++){
			test_mysgemm_v9(m, n, k, alpha, dA, dB, beta, dC);
		}

        /*
        DPCT1012:5: Detected kernel execution time measurement pattern and
        generated an initial code for time measurements in SYCL. You can change
        the way time is measured depending on your goals.
        */
        end_ct1 = std::chrono::steady_clock::now();
        elapsed_time =
            std::chrono::duration<float, std::milli>(end_ct1 - beg_ct1).count();
        elapsed_time /= 1000.;

        printf("Average elapsed time: %f second, performance: %f GFLOPS.\n", elapsed_time/N,2.*1e-9*N*m*n*k/elapsed_time);
        fflush(stdout);
        copy_matrix(C_ref,C,m*n);//sync C with cuBLAS to prepare for the next run
    }
    hipDeviceSynchronize();
    free(A);free(B);free(C);free(C_ref);
    hipFree(dA);hipFree(dB);hipFree(dC);hipFree(dC_ref);
    hipDeviceSynchronize();
    return 0;
}

void randomize_matrix(FLOAT* mat, int N){
    srand(time(NULL)); int i;
    for (i = 0; i < N; i++) {
        FLOAT tmp = (float)(rand() % 5) + 0.01 * (rand() % 5);
        tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
        //tmp = i;
        mat[i] = tmp;
    }
}

void copy_matrix(FLOAT *src, FLOAT *dest, int n){
    int i;
    for (i = 0; src + i && dest + i && i < n; i++) *(dest + i) = *(src + i);
    if (i != n) printf("copy failed at %d while there are %d elements in total.\n", i, n);
}

void test_mysgemm_v9(INT M, INT N, INT K, FLOAT alpha, FLOAT *A, FLOAT *B,
                     FLOAT beta, FLOAT *C) {
    hipDeviceSynchronize();
    dim3 blockDim(512);
    dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(M, 128));
    mysgemm_v9<128,16,128,16,64,4,1,8,7,3,3,3,512,9,2,3,2,4,4,2,6,3,5,31,2,7,7,2048,2048,1,2,8,2,0><<<gridDim, blockDim>>>(M,N,K,alpha,A,B,beta,C);
    hipDeviceSynchronize();
}

void test_mysgemm_v1(INT M, INT N, INT K, FLOAT alpha, FLOAT *A, FLOAT *B,
                     FLOAT beta, FLOAT *C) {
    hipDeviceSynchronize();
    dim3 blockDim(32, 32);
    dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 32));
    mysgemm_v1<<<gridDim, blockDim>>>(M,N,K,alpha,A,B,beta,C);
    hipDeviceSynchronize();
}


bool verify_matrix(FLOAT *mat1, FLOAT *mat2, int n){
    double diff = 0.0;
    int i;
    for (i = 0; mat1 + i && mat2 + i && i < n; i++){
        diff = fabs( (double)mat1[i] - (double)mat2[i] );
        if (diff > 1e-2) {
            printf("error. %5.2f,%5.2f,%d\n", mat1[i],mat2[i],i);
            return false;
        }
    }
    return true;
}