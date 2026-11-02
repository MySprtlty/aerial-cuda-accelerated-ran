// gc_devgraph: device-side graph launch (what pusch_deviceGraphLaunchEn=1 uses) from a kernel running in a green-context stream.
// Two graphs: one captured on the green stream, one built with cudaGraphAddKernelNode; each instantiated with DeviceLaunch
// and fired from a launcher kernel running on the green-context stream (cudaGraphLaunch device API, fire-and-forget).
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#define CK(x) do{CUresult r=(x); if(r!=CUDA_SUCCESS){const char*s; cuGetErrorString(r,&s); printf("%s failed: %s (%d)\n",#x,s,(int)r); return 1;}}while(0)
__global__ void smid_kernel(unsigned* hist, int spin_us){
    unsigned smid; asm("mov.u32 %0, %%smid;" : "=r"(smid));
    if(threadIdx.x==0) atomicAdd(&hist[smid],1u);
    long long t0=clock64(); while(clock64()-t0 < (long long)spin_us*1980) {}
}
__global__ void launcher(cudaGraphExec_t ge){ if(threadIdx.x==0) cudaGraphLaunch(ge, cudaStreamGraphFireAndForget); }
int main(){
    CK(cuInit(0)); CUdevice dev; CK(cuDeviceGet(&dev,0)); int nsm=0; CK(cuDeviceGetAttribute(&nsm,CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,dev));
    int mps=0; cuDeviceGetAttribute(&mps,CU_DEVICE_ATTRIBUTE_MPS_ENABLED,dev); printf("device SMs %d, MPS enabled %d\n",nsm,mps);
    CUcontext prim; CK(cuDevicePrimaryCtxRetain(&prim,dev)); CK(cuCtxSetCurrent(prim));
    CUdevResource all={}, A={}, rest={}; unsigned ng=1; CK(cuDeviceGetDevResource(dev,&all,CU_DEV_RESOURCE_TYPE_SM)); CK(cuDevSmResourceSplitByCount(&A,&ng,&all,&rest,0,32));
    CUdevResourceDesc dA; CK(cuDevResourceGenerateDesc(&dA,&A,1)); CUgreenCtx gA; CK(cuGreenCtxCreate(&gA,dA,dev,CU_GREEN_CTX_DEFAULT_STREAM)); CUcontext cA; CK(cuCtxFromGreenCtx(&cA,gA));
    CK(cuCtxSetCurrent(cA)); cudaStream_t sA; cudaStreamCreateWithPriority(&sA,cudaStreamNonBlocking,-2);
    unsigned* hist; cudaMallocManaged(&hist,256*sizeof(unsigned));
    auto report=[&](const char* name, cudaError_t e){ cudaDeviceSynchronize(); std::vector<int> used; for(int i=0;i<nsm;i++) if(hist[i]) used.push_back(i); printf("%s: %s, ran on %zu distinct SMs [%d..%d]\n",name,cudaGetErrorString(e),used.size(),used.empty()?-1:used.front(),used.empty()?-1:used.back()); };
    // graph 1: captured on sA (ctx A current)
    cudaGraph_t g1; cudaGraphExec_t ge1; cudaStreamBeginCapture(sA,cudaStreamCaptureModeGlobal); smid_kernel<<<1024,64,0,sA>>>(hist,200); cudaStreamEndCapture(sA,&g1);
    cudaError_t e1=cudaGraphInstantiateWithFlags(&ge1,g1,cudaGraphInstantiateFlagDeviceLaunch); if(e1){printf("inst1 %s\n",cudaGetErrorString(e1)); return 1;}
    cudaGraphUpload(ge1,sA); cudaStreamSynchronize(sA);
    cudaMemset(hist,0,256*4); cudaDeviceSynchronize(); launcher<<<1,32,0,sA>>>(ge1); report("(6a) captured-on-A graph, device-launched from kernel on stream A", cudaGetLastError());
    // graph 2: hand-built kernel node (PRACH style), instantiated with ctx A current
    cudaGraph_t g2; cudaGraphExec_t ge2; cudaGraphCreate(&g2,0); cudaKernelNodeParams kp={}; void* args[2]={&hist,nullptr}; int spin=200; args[1]=&spin;
    kp.func=(void*)smid_kernel; kp.gridDim=dim3(1024); kp.blockDim=dim3(64); kp.kernelParams=args; cudaGraphNode_t n; cudaGraphAddKernelNode(&n,g2,nullptr,0,&kp);
    cudaError_t e2=cudaGraphInstantiateWithFlags(&ge2,g2,cudaGraphInstantiateFlagDeviceLaunch); if(e2){printf("inst2 %s\n",cudaGetErrorString(e2)); return 1;}
    cudaGraphUpload(ge2,sA); cudaStreamSynchronize(sA);
    cudaMemset(hist,0,256*4); cudaDeviceSynchronize(); launcher<<<1,32,0,sA>>>(ge2); report("(6b) hand-built graph, device-launched from kernel on stream A", cudaGetLastError());
    cudaMemset(hist,0,256*4); cudaDeviceSynchronize(); cudaError_t e3=cudaGraphLaunch(ge2,sA); report("(6c) hand-built graph, host-launched into stream A", e3);
    cudaMemset(hist,0,256*4); cudaDeviceSynchronize(); cudaError_t e4=cudaGraphLaunch(ge1,sA); report("(6d) captured graph, host-launched into stream A", e4);
    // control tests for the "construction-time context" hypothesis and for device launch under the primary context
    CK(cuCtxSetCurrent(prim)); cudaStream_t sP; cudaStreamCreateWithFlags(&sP,cudaStreamNonBlocking);
    cudaGraph_t g3; cudaGraphExec_t ge3; cudaGraphCreate(&g3,0); cudaGraphNode_t n3; cudaGraphAddKernelNode(&n3,g3,nullptr,0,&kp); // nodes added under PRIMARY
    CK(cuCtxSetCurrent(cA)); cudaGraphInstantiate(&ge3,g3,NULL,NULL,0);                                                           // instantiated under A
    cudaMemset(hist,0,256*4); cudaDeviceSynchronize(); cudaError_t e5=cudaGraphLaunch(ge3,sA); report("(6e) nodes added under primary, instantiated under A, launched into stream A", e5);
    CK(cuCtxSetCurrent(cA)); cudaGraph_t g4; cudaGraphExec_t ge4; cudaGraphCreate(&g4,0); cudaGraphNode_t n4; cudaGraphAddKernelNode(&n4,g4,nullptr,0,&kp); // nodes added under A
    CK(cuCtxSetCurrent(prim)); cudaGraphInstantiate(&ge4,g4,NULL,NULL,0);                                                          // instantiated under primary
    cudaMemset(hist,0,256*4); cudaDeviceSynchronize(); cudaError_t e6=cudaGraphLaunch(ge4,sP); report("(6g) nodes added under A, instantiated under primary, launched into primary stream", e6);
    // device launch entirely under the primary context
    CK(cuCtxSetCurrent(prim)); cudaGraph_t g5; cudaGraphExec_t ge5; cudaGraphCreate(&g5,0); cudaGraphNode_t n5; cudaGraphAddKernelNode(&n5,g5,nullptr,0,&kp);
    cudaError_t e7=cudaGraphInstantiateWithFlags(&ge5,g5,cudaGraphInstantiateFlagDeviceLaunch); cudaGraphUpload(ge5,sP); cudaStreamSynchronize(sP);
    cudaMemset(hist,0,256*4); cudaDeviceSynchronize(); launcher<<<1,32,0,sP>>>(ge5); report("(6f) device-launched graph, launcher kernel on a PRIMARY stream", cudaGetLastError());
    // device launch: graph built+instantiated under primary, launcher kernel on stream A
    cudaMemset(hist,0,256*4); cudaDeviceSynchronize(); CK(cuCtxSetCurrent(cA)); launcher<<<1,32,0,sA>>>(ge5); report("(6h) primary-built device-launch graph, launcher kernel on stream A", cudaGetLastError());
    return 0;
}
