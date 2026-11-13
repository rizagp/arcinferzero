// Real imported expert: independent source F32 dump vs CPU decoder vs GPU vblock.
#include "../zero/clrt.h"
#include "../src/quants.h"
#include <fstream>
#include <cstdio>
int main(int argc,char** argv) {
  if(argc!=5)return 2; // expert file, source gate/up/down F32 dumps
  ClRt cl;if(!cl.init())return 1;
  const std::string source=read_file("zero/kernels/ar4.cl")+read_file("zero/kernels/decode.cl")+R"(
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void ar4_check(__global const uchar* raw,int N,int K,__global const uint* tabs,__global float* out){
  __local uint table[VB_TABLES];vb_tables(table,tabs);
  int row=get_global_id(0),block=get_global_id(1),s=block/8,ib=block%8;
  if(row>=N)return;uint w[8];float d,dm;int sc0,sc1,m0,m1;
  vblock(raw+((size_t)row*(K/256)+s)*BBP,ib,table,w,&d,&dm,&sc0,&sc1,&m0,&m1);
  for(int j=0;j<32;++j)out[(size_t)row*K+block*32+j]=d*(float)(char)(w[j/4]>>(8*(j%4)));
})";
  cl_program program=cl.build(source,"-cl-std=CL3.0 -DQT="+std::to_string(GGML_AR4_128));
  cl_kernel kernel=cl.kernel(program,"ar4_check");
  std::ifstream input(argv[1],std::ios::binary);input.seekg(4096);
  size_t checked=0;
  for(int part=0;part<3;++part){
    const int N=part==2?2560:640,K=part==2?640:2560,Kp=padded_k(K);
    const size_t rowbytes=(size_t)Kp/256*136;
    std::vector<uint8_t> raw((size_t)N*rowbytes);input.read((char*)raw.data(),raw.size());if(!input)return 1;
    std::vector<float> golden((size_t)N*K),want((size_t)N*Kp),got(want.size());
    std::ifstream ref(argv[part+2],std::ios::binary);ref.read((char*)golden.data(),golden.size()*4);if(!ref)return 1;
    for(int row=0;row<N;++row){std::vector<uint8_t> original((size_t)Kp/128*66);
      for(int s=0;s<Kp/256;++s){const uint8_t* p=raw.data()+row*rowbytes+s*136;uint8_t* q=original.data()+s*132;
        memcpy(q,p,2);memcpy(q+2,p+4,64);memcpy(q+66,p+2,2);memcpy(q+68,p+68,64);}
      dequant_row(GGML_AR4_128,original.data(),Kp,want.data()+(size_t)row*Kp);
      if(memcmp(want.data()+(size_t)row*Kp,golden.data()+(size_t)row*K,K*4))return 1;
    }
    auto weights=cl.buffer(CL_MEM_READ_ONLY,raw.size(),raw.data());
    const auto table=build_iq_tables();auto tabs=cl.buffer(CL_MEM_READ_ONLY,table.size()*4,table.data());
    auto output=cl.buffer(CL_MEM_READ_WRITE,got.size()*4);
    clSetKernelArg(kernel,0,sizeof(weights),&weights);clSetKernelArg(kernel,1,4,&N);clSetKernelArg(kernel,2,4,&Kp);
    clSetKernelArg(kernel,3,sizeof(tabs),&tabs);clSetKernelArg(kernel,4,sizeof(output),&output);
    const size_t global[2]={(size_t)(N+63)/64*64,(size_t)Kp/32},local[2]={64,1};
    CL_CHECK(clEnqueueNDRangeKernel(cl.queue,kernel,2,nullptr,global,local,0,nullptr,nullptr));
    CL_CHECK(clEnqueueReadBuffer(cl.queue,output,CL_TRUE,0,got.size()*4,got.data(),0,nullptr,nullptr));
    if(memcmp(want.data(),got.data(),got.size()*4))return 1;
    checked+=golden.size();printf("part=%d source/CPU/GPU bit-identical (%zu weights)\n",part,golden.size());fflush(stdout);
    for(auto b:{weights,tabs,output})clReleaseMemObject(b);
  }
  printf("PASS: %zu real source weights, including down padding\n",checked);return 0;
}
