// gc_bench: measures (1) cuGreenCtxCreate/cuGreenCtxDestroy cost, (2) stream-switch cost between green contexts,
// (3) which SMs a kernel actually runs on when launched into a stream of a green context (%smid histogram),
// (4) overlapping partitions (two green contexts sharing SMs) and a kernel launched after cuCtxSetCurrent to another ctx.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include <vector>
#include <algorithm>
#define CK(x) do{CUresult r=(x); if(r!=CUDA_SUCCESS){const char*s; cuGetErrorString(r,&s); printf("%s failed: %s (%d)\n",#x,s,(int)r); return 1;}}while(0)
__global__ void smid_kernel(unsigned* hist, int spin_us){
    unsigned smid; asm("mov.u32 %0, %%smid;" : "=r"(smid));
    if(threadIdx.x==0) atomicAdd(&hist[smid],1u);
    long long t0=clock64(); while(clock64()-t0 < (long long)spin_us*1980) {}
}
static double now_us(){ return std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int main(){
    CK(cuInit(0)); CUdevice dev; CK(cuDeviceGet(&dev,0));
    int nsm=0; CK(cuDeviceGetAttribute(&nsm,CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,dev));
    int mps=0; cuDeviceGetAttribute(&mps,CU_DEVICE_ATTRIBUTE_MPS_ENABLED,dev);
    printf("device SMs %d, MPS enabled %d\n", nsm, mps);
    CUcontext prim; CK(cuDevicePrimaryCtxRetain(&prim,dev)); CK(cuCtxSetCurrent(prim));
    CUdevResource all={}; CK(cuDeviceGetDevResource(dev,&all,CU_DEV_RESOURCE_TYPE_SM));
    // --- (1) creation/destroy cost for a 16-SM partition, 20 repetitions
    double tc=0,td=0; const int REP=20;
    for(int i=0;i<REP;i++){
        CUdevResource part={}, rem={}; unsigned ng=1;
        CK(cuDevSmResourceSplitByCount(&part,&ng,&all,&rem,0,16));
        CUdevResourceDesc d; CK(cuDevResourceGenerateDesc(&d,&part,1));
        double t0=now_us(); CUgreenCtx g; CK(cuGreenCtxCreate(&g,d,dev,CU_GREEN_CTX_DEFAULT_STREAM)); double t1=now_us();
        CUstream s; CK(cuGreenCtxStreamCreate(&s,g,CU_STREAM_NON_BLOCKING,0)); CK(cuStreamDestroy(s));
        double t2=now_us(); CK(cuGreenCtxDestroy(g)); double t3=now_us();
        if(i>0){tc+=t1-t0; td+=t3-t2;}
    }
    printf("(1) green ctx create avg %.1f us, destroy avg %.1f us (16-SM partition, %d reps, first excluded)\n", tc/(REP-1), td/(REP-1), REP);
    // --- (2)+(3): two disjoint partitions A (first split result) and B (remainder re-split), plus an overlapping C
    CUdevResource A={}, restA={}; unsigned ng=1; CK(cuDevSmResourceSplitByCount(&A,&ng,&all,&restA,0,32));
    // a remainder resource cannot be re-split directly (CUDA_ERROR 915): cuphydriver creates a temporary green context from it
    // and re-splits the resource queried from that context (tmpGreenContextsForResplit in context.cpp). Same here.
    CUdevResource B={}, restB={}; ng=1;
    { CUdevResourceDesc dT; CK(cuDevResourceGenerateDesc(&dT,&restA,1)); CUgreenCtx gT; CK(cuGreenCtxCreate(&gT,dT,dev,CU_GREEN_CTX_DEFAULT_STREAM));
      CUdevResource restA2={}; CK(cuGreenCtxGetDevResource(gT,&restA2,CU_DEV_RESOURCE_TYPE_SM));
      CUresult rr=cuDevSmResourceSplitByCount(&B,&ng,&restA2,&restB,0,24);
      printf("re-split of the remainder via a temp green ctx (%u SMs): %s\n", restA2.sm.smCount, rr==CUDA_SUCCESS?"ok":"FAILED");
      if(rr!=CUDA_SUCCESS) return 1; CK(cuGreenCtxDestroy(gT)); }
    CUdevResource C={}, restC={}; ng=1; CK(cuDevSmResourceSplitByCount(&C,&ng,&all,&restC,0,48)); // overlaps A (both start from 'all')
    printf("partitions: A %u SMs, B %u SMs (from remainder %u), C %u SMs (from full set, overlaps A)\n", A.sm.smCount, B.sm.smCount, restA.sm.smCount, C.sm.smCount);
    CUdevResourceDesc dA,dB,dC; CK(cuDevResourceGenerateDesc(&dA,&A,1)); CK(cuDevResourceGenerateDesc(&dB,&B,1)); CK(cuDevResourceGenerateDesc(&dC,&C,1));
    CUgreenCtx gA,gB,gC; CK(cuGreenCtxCreate(&gA,dA,dev,CU_GREEN_CTX_DEFAULT_STREAM)); CK(cuGreenCtxCreate(&gB,dB,dev,CU_GREEN_CTX_DEFAULT_STREAM)); CK(cuGreenCtxCreate(&gC,dC,dev,CU_GREEN_CTX_DEFAULT_STREAM));
    CUcontext cA,cB,cC; CK(cuCtxFromGreenCtx(&cA,gA)); CK(cuCtxFromGreenCtx(&cB,gB)); CK(cuCtxFromGreenCtx(&cC,gC));
    // streams created the cuphydriver way: cudaStreamCreateWithPriority while the converted context is current
    cudaStream_t sA,sB,sC; CK(cuCtxSetCurrent(cA)); cudaStreamCreateWithPriority(&sA,cudaStreamNonBlocking,-2);
    CK(cuCtxSetCurrent(cB)); cudaStreamCreateWithPriority(&sB,cudaStreamNonBlocking,-2);
    CK(cuCtxSetCurrent(cC)); cudaStreamCreateWithPriority(&sC,cudaStreamNonBlocking,-2);
    unsigned *hist; cudaMallocManaged(&hist, 256*sizeof(unsigned));
    auto run=[&](const char* name, cudaStream_t s, CUcontext cur){
        cudaMemset(hist,0,256*sizeof(unsigned)); cudaDeviceSynchronize();
        CK(cuCtxSetCurrent(cur));
        smid_kernel<<<1024,64,0,s>>>(hist,200); cudaError_t e=cudaGetLastError(); if(e){printf("%s launch err %s\n",name,cudaGetErrorString(e)); return 1;}
        cudaStreamSynchronize(s);
        std::vector<int> used; for(int i=0;i<nsm;i++) if(hist[i]) used.push_back(i);
        printf("%s: kernel ran on %zu distinct SMs [%d..%d]\n", name, used.size(), used.empty()?-1:used.front(), used.empty()?-1:used.back());
        return 0;
    };
    if(run("(3a) stream of A, ctx A current", sA, cA)) return 1;
    if(run("(3b) stream of B, ctx B current", sB, cB)) return 1;
    if(run("(3c) stream of C (overlaps A), ctx C current", sC, cC)) return 1;
    if(run("(3d) stream of A but ctx B current (stream decides?)", sA, cB)) return 1;
    if(run("(3e) stream of B but primary ctx current", sB, prim)) return 1;
    // --- (2) switch cost: alternate launches A/B 2000 times, measure host-side launch + ctx switch
    CK(cuCtxSetCurrent(cA)); smid_kernel<<<1,32,0,sA>>>(hist,1); CK(cuCtxSetCurrent(cB)); smid_kernel<<<1,32,0,sB>>>(hist,1); cudaDeviceSynchronize();
    double t0=now_us(); for(int i=0;i<2000;i++){ CUcontext c=(i&1)?cB:cA; cudaStream_t s=(i&1)?sB:sA; cuCtxSetCurrent(c); smid_kernel<<<1,32,0,s>>>(hist,1);} double t1=now_us(); cudaDeviceSynchronize();
    printf("(2) alternating ctx switch + launch: %.2f us per launch (host side, 2000 launches)\n", (t1-t0)/2000);
    // --- (4) events across green contexts: record on sA, wait on sB
    cudaEvent_t ev; CK(cuCtxSetCurrent(cA)); cudaEventCreateWithFlags(&ev,cudaEventDisableTiming);
    smid_kernel<<<64,64,0,sA>>>(hist,100); cudaEventRecord(ev,sA);
    CK(cuCtxSetCurrent(cB)); cudaError_t e=cudaStreamWaitEvent(sB,ev,0); smid_kernel<<<64,64,0,sB>>>(hist,1); cudaError_t e2=cudaStreamSynchronize(sB);
    printf("(4) event recorded in ctx A, waited in ctx B: wait=%s sync=%s\n", cudaGetErrorString(e), cudaGetErrorString(e2));
    // --- (5) CUDA graph (PRACH uses one: prach_receiver.cu cudaGraphInstantiate once, cudaGraphLaunch(graph, strm) per slot):
    //         instantiate once while the primary context is current, then launch the same exec graph into sA and sB.
    {
        CK(cuCtxSetCurrent(prim));
        cudaStream_t cap; cudaStreamCreateWithFlags(&cap,cudaStreamNonBlocking);
        cudaGraph_t g; cudaGraphExec_t ge;
        cudaStreamBeginCapture(cap,cudaStreamCaptureModeGlobal); smid_kernel<<<1024,64,0,cap>>>(hist,200); cudaStreamEndCapture(cap,&g);
        cudaError_t ei=cudaGraphInstantiate(&ge,g,NULL,NULL,0); if(ei){printf("(5) graph instantiate err %s\n",cudaGetErrorString(ei)); return 1;}
        auto rung=[&](const char* name, cudaStream_t s){
            cudaMemset(hist,0,256*sizeof(unsigned)); cudaDeviceSynchronize();
            cudaError_t el=cudaGraphLaunch(ge,s); cudaStreamSynchronize(s);
            std::vector<int> used; for(int i=0;i<nsm;i++) if(hist[i]) used.push_back(i);
            printf("%s: launch=%s, ran on %zu distinct SMs [%d..%d]\n", name, cudaGetErrorString(el), used.size(), used.empty()?-1:used.front(), used.empty()?-1:used.back());
        };
        rung("(5a) exec graph -> stream of A (32)", sA);
        rung("(5b) same exec graph -> stream of B (24)", sB);
        rung("(5c) same exec graph -> primary stream", cap);
        // variants: instantiate while green ctx A is current (what PhyPrachAggr::createPhyObj does under setCtx()), and capture on sA
        CK(cuCtxSetCurrent(cA));
        cudaGraphExec_t geA; cudaError_t eA=cudaGraphInstantiate(&geA,g,NULL,NULL,0);
        if(eA){printf("(5d) instantiate under ctx A err %s\n",cudaGetErrorString(eA));}
        else { cudaGraphExec_t keep=ge; ge=geA; rung("(5d) graph instantiated under ctx A -> stream of A (32)", sA); rung("(5e) graph instantiated under ctx A -> stream of B (24)", sB); ge=keep; }
        cudaGraph_t g2; cudaGraphExec_t ge2; CK(cuCtxSetCurrent(cA));
        cudaError_t ec=cudaStreamBeginCapture(sA,cudaStreamCaptureModeGlobal); smid_kernel<<<1024,64,0,sA>>>(hist,200); cudaError_t ee=cudaStreamEndCapture(sA,&g2);
        if(ec||ee){printf("(5f) capture on green stream: begin=%s end=%s\n",cudaGetErrorString(ec),cudaGetErrorString(ee));}
        else { cudaError_t ei2=cudaGraphInstantiate(&ge2,g2,NULL,NULL,0); if(ei2) printf("(5f) instantiate err %s\n",cudaGetErrorString(ei2)); else { cudaGraphExec_t keep=ge; ge=ge2; rung("(5f) graph captured on stream of A, instantiated under ctx A -> stream of A", sA); rung("(5g) same -> stream of B (24)", sB); ge=keep; } }
        // plain kernel again on sA, to show the non-graph path is still confined after the graph launches
        CK(cuCtxSetCurrent(prim)); cudaMemset(hist,0,256*sizeof(unsigned)); cudaDeviceSynchronize(); smid_kernel<<<1024,64,0,sA>>>(hist,200); cudaStreamSynchronize(sA);
        { std::vector<int> used; for(int i=0;i<nsm;i++) if(hist[i]) used.push_back(i); printf("(5h) plain kernel -> stream of A again: %zu distinct SMs\n", used.size()); }
    }
    CK(cuGreenCtxDestroy(gA)); CK(cuGreenCtxDestroy(gB)); CK(cuGreenCtxDestroy(gC));
    printf("done\n"); return 0;
}
