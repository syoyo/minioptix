#include <cstdio>
#include <iomanip>
#include <iostream>

#ifdef MINIOPTIX_USE_CUEW
#include <cuew.h>

// Disable including cuda.h in optix.h
#define OPTIX_DONT_INCLUDE_CUDA

#endif

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#define CU_CHECK(cond)                                                 \
  do {                                                                 \
    CUresult ret = cond;                                               \
    if (ret != CUDA_SUCCESS) {                                         \
      std::cerr << __FILE__ << ":" << __LINE__                         \
                << " CUDA Device API failed. retcode " << ret << "\n"; \
      exit(-1);                                                        \
    }                                                                  \
  } while (0)

#define OPTIX_CHECK(callfun)                                                \
  do {                                                                      \
    OptixResult ret = callfun;                                              \
    if (ret != OPTIX_SUCCESS) {                                             \
      std::cerr << __FILE__ << ":" << __LINE__ << " Optix call" << #callfun \
                << " failed. retcode " << ret << "\n";                      \
      exit(-1);                                                             \
    }                                                                       \
  } while (0)

#define OPTIX_CHECK_LOG(callfun)                                               \
  do {                                                                         \
    OptixResult ret = callfun;                                                 \
    const size_t sizeof_log_returned = sizeof_log;                             \
    sizeof_log = sizeof(logbuf); /* reset sizeof_log for future calls */       \
    if (ret != OPTIX_SUCCESS) {                                                \
      std::cerr << __FILE__ << ":" << __LINE__ << " Optix call" << #callfun    \
                << " failed. log: " << logbuf                                  \
                << (sizeof_log_returned > sizeof(logbuf) ? "<TRUNCATED>" : "") \
                << "\n";                                                       \
      exit(-1);                                                                \
    }                                                                          \
  } while (0)

struct RayGenData {
  float r, g, b;
};

struct Params {
  uint8_t* image;  // RGBA
  unsigned int image_width;
};

template <typename T>
struct SbtRecord {
  // TODO: MSVC align
  char header[OPTIX_SBT_RECORD_HEADER_SIZE]
      __attribute__((aligned(OPTIX_SBT_RECORD_ALIGNMENT)));
  T data;
};

typedef SbtRecord<RayGenData> RayGenSbtRecord;
typedef SbtRecord<int> MissSbtRecord;
typedef SbtRecord<int> HitGroupSbtRecord;

bool CUDAAllocDeviceMem(CUdeviceptr* dptr, size_t sz) {
  CU_CHECK(cuMemAlloc(dptr, sz));

  return true;
}

static void context_log_cb(unsigned int level, const char* tag,
                           const char* message, void* /*cbdata */) {
  std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag
            << "]: " << message << "\n";
}

int main(int argc, char** argv) {
#ifdef MINIOPTIX_USE_CUEW
  if (cuewInit(CUEW_INIT_CUDA) != CUEW_SUCCESS) {
    std::cerr << "Failed to initialize CUDA\n";
    return -1;
  }

  printf("CUDA compiler path: %s, compiler version: %d\n", cuewCompilerPath(),
         cuewCompilerVersion());

  // Currently we require NVRTC to be available for runtime .cu compilation.
  if (cuewInit(CUEW_INIT_NVRTC) != CUEW_SUCCESS) {
    std::cerr << "Failed to initialize NVRTC. NVRTC library is not available "
                 "or not found in the system search path\n";
    return -1;
  } else {
    int major, minor;
    nvrtcVersion(&major, &minor);
    std::cout << "Found NVRTC runtime compilation library version " << major
              << "." << minor << "\n";
  }
#endif

  if (cuInit(0) != CUDA_SUCCESS) {
    std::cerr << "Failed to init CUDA\n";
    return -1;
  }

  //
  // Initialize CUDA and create OptiX context
  //
  OptixDeviceContext context = nullptr;
  CUcontext cuCtx{};
  {
    // // Initialize CUDA
    // if (cudaFree( 0 ) != cudaSucces) {
    //   LDLOG_ERROR("Failed to initialize CUDA");
    //   return -1;
    // }
    //
    int counts{0};
    // if (CUDA_SUCCESS != cuDeviceGetCount(&counts)) {
    //  LDLOG_ERROR("Failed to get CUDA device count");
    //  return -1;
    //}
    CU_CHECK(cuDeviceGetCount(&counts));

    std::cout << "# of CUDA devices: " << counts << "\n";
    if (counts < 1) {
      std::cerr << "No CUDA device found\n";
      return -1;
    }

    CUdevice device{};
    if (CUDA_SUCCESS != cuDeviceGet(&device, /* devid */ 0)) {
      std::cerr << "Failed to get CUDA device.\n";
      return -1;
    }

    {
      int major, minor;
      CU_CHECK(cuDeviceComputeCapability(&major, &minor, device));
      std::cerr << "compute capability: " << major << "." << minor << "\n";
    }

    if (CUDA_SUCCESS != cuCtxCreate(&cuCtx, /* flags */ 0, device)) {
      std::cerr << "Failed to get Create CUDA context.\n";
      return -1;
    }

    // CUmodule   module;
    // CUresult ret = cuModuleLoadData(&module, reinterpret_cast<const void
    // *>(ptx_code)); std::cout << "ret = " << ret << "\n";

    // CUfunction func;
    // if (cuModuleGetFunction(&func, module, "_Z3addffPf") != CUDA_SUCCESS) {
    //  LDLOG_ERROR("Failed to get a function");
    //  cuCtxDetach(cuCtx);
    //  exit(-1);
    //}
  }

  // === OptiX ============================================

  {
    if (optixInit() != OPTIX_SUCCESS) {
      std::cerr << "Failed to initialize OptiX\n";
      return -1;
    }

    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &context_log_cb;
    options.logCallbackLevel = 4;
    OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));

    char logbuf[2048];

    //
    // Create module
    //
    OptixModule module = nullptr;
    OptixPipelineCompileOptions pipeline_compile_options = {};
    {
      OptixModuleCompileOptions module_compile_options = {};
      module_compile_options.maxRegisterCount =
          OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
      module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
      module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

      pipeline_compile_options.usesMotionBlur = false;
      pipeline_compile_options.traversableGraphFlags =
          OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
      pipeline_compile_options.numPayloadValues = 2;
      pipeline_compile_options.numAttributeValues = 2;
      pipeline_compile_options.exceptionFlags =
          OPTIX_EXCEPTION_FLAG_NONE;  // TODO: should be
                                      // OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
      pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

      const std::string ptx;  // = sutil::getPtxString(OPTIX_SAMPLE_NAME,
                              // OPTIX_SAMPLE_DIR, "draw_solid_color.cu");
      size_t sizeof_log = sizeof(logbuf);

      OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
          context, &module_compile_options, &pipeline_compile_options,
          ptx.c_str(), ptx.size(), logbuf, &sizeof_log, &module));
    }

    //
    // Create program groups, including NULL miss and hitgroups
    //
    OptixProgramGroup raygen_prog_group = nullptr;
    OptixProgramGroup miss_prog_group = nullptr;
    OptixProgramGroup hitgroup_prog_group = nullptr;
    {
      OptixProgramGroupOptions program_group_options =
          {};  // Initialize to zeros

      OptixProgramGroupDesc raygen_prog_group_desc = {};  //
      raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
      raygen_prog_group_desc.raygen.module = module;
      raygen_prog_group_desc.raygen.entryFunctionName =
          "__raygen__draw_solid_color";
      size_t sizeof_log = sizeof(logbuf);
      OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &raygen_prog_group_desc,
                                              1,  // num program groups
                                              &program_group_options, logbuf,
                                              &sizeof_log, &raygen_prog_group));

      // Leave miss group's module and entryfunc name null
      OptixProgramGroupDesc miss_prog_group_desc = {};
      miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
      sizeof_log = sizeof(logbuf);
      OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &miss_prog_group_desc,
                                              1,  // num program groups
                                              &program_group_options, logbuf,
                                              &sizeof_log, &miss_prog_group));

      // Leave hit group's module and entryfunc name null
      OptixProgramGroupDesc hitgroup_prog_group_desc = {};
      hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
      sizeof_log = sizeof(logbuf);
      OPTIX_CHECK_LOG(optixProgramGroupCreate(
          context, &hitgroup_prog_group_desc,
          1,  // num program groups
          &program_group_options, logbuf, &sizeof_log, &hitgroup_prog_group));
    }

    //
    // Link pipeline
    //
    OptixPipeline pipeline = nullptr;
    {
      const uint32_t max_trace_depth = 0;
      OptixProgramGroup program_groups[] = {raygen_prog_group};

      OptixPipelineLinkOptions pipeline_link_options = {};
      pipeline_link_options.maxTraceDepth = max_trace_depth;
      pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
      size_t sizeof_log = sizeof(logbuf);
      OPTIX_CHECK_LOG(optixPipelineCreate(
          context, &pipeline_compile_options, &pipeline_link_options,
          program_groups, sizeof(program_groups) / sizeof(program_groups[0]),
          logbuf, &sizeof_log, &pipeline));

      OptixStackSizes stack_sizes = {};
      for (auto& prog_group : program_groups) {
        OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes));
      }

      uint32_t direct_callable_stack_size_from_traversal;
      uint32_t direct_callable_stack_size_from_state;
      uint32_t continuation_stack_size;
      OPTIX_CHECK(optixUtilComputeStackSizes(
          &stack_sizes, max_trace_depth,
          0,  // maxCCDepth
          0,  // maxDCDEpth
          &direct_callable_stack_size_from_traversal,
          &direct_callable_stack_size_from_state, &continuation_stack_size));
      OPTIX_CHECK(optixPipelineSetStackSize(
          pipeline, direct_callable_stack_size_from_traversal,
          direct_callable_stack_size_from_state, continuation_stack_size,
          2  // maxTraversableDepth
          ));
    }

    //
    // Set up shader binding table
    //
    OptixShaderBindingTable sbt = {};
    {
      CUdeviceptr raygen_record;
      const size_t raygen_record_size = sizeof(RayGenSbtRecord);
      // CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &raygen_record ),
      // raygen_record_size ) );
      CUDAAllocDeviceMem(&raygen_record, raygen_record_size);
      RayGenSbtRecord rg_sbt;
      OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group, &rg_sbt));
      rg_sbt.data = {0.462f, 0.725f, 0.f};

      // CUDA_CHECK( cudaMemcpy(
      //            reinterpret_cast<void*>( raygen_record ),
      //            &rg_sbt,
      //            raygen_record_size,
      //            cudaMemcpyHostToDevice
      //            ) );

      CUdeviceptr miss_record;
      size_t miss_record_size = sizeof(MissSbtRecord);
      // CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &miss_record ),
      // miss_record_size ) );
      CUDAAllocDeviceMem(&miss_record, miss_record_size);
      RayGenSbtRecord ms_sbt;
      OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_group, &ms_sbt));

      // CUDA_CHECK( cudaMemcpy(
      //            reinterpret_cast<void*>( miss_record ),
      //            &ms_sbt,
      //            miss_record_size,
      //            cudaMemcpyHostToDevice
      //            ) );

      CUdeviceptr hitgroup_record;
      size_t hitgroup_record_size = sizeof(HitGroupSbtRecord);
      // CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &hitgroup_record ),
      // hitgroup_record_size ) );
      CUDAAllocDeviceMem(&hitgroup_record, hitgroup_record_size);
      RayGenSbtRecord hg_sbt;
      OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_group, &hg_sbt));
      // CUDA_CHECK( cudaMemcpy(
      //            reinterpret_cast<void*>( hitgroup_record ),
      //            &hg_sbt,
      //            hitgroup_record_size,
      //            cudaMemcpyHostToDevice
      //            ) );

      sbt.raygenRecord = raygen_record;
      sbt.missRecordBase = miss_record;
      sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
      sbt.missRecordCount = 1;
      sbt.hitgroupRecordBase = hitgroup_record;
      sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
      sbt.hitgroupRecordCount = 1;
    }

    int width = 512;
    int height = 512;

    //
    // launch
    //
    {
      CUstream stream;
      // CUDA_CHECK( cudaStreamCreate( &stream ) );
      CU_CHECK(cuStreamCreate(&stream, CU_STREAM_DEFAULT));

      Params params;
      // params.image       = output_buffer.map();
      // params.image_width = width;

      CUdeviceptr d_param;
      // CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_param ), sizeof(
      // Params ) ) ); CUDA_CHECK( cudaMemcpy(
      //            reinterpret_cast<void*>( d_param ),
      //            &params, sizeof( params ),
      //            cudaMemcpyHostToDevice
      //            ) );

      OPTIX_CHECK(optixLaunch(pipeline, stream, d_param, sizeof(Params), &sbt,
                              width, height, /*depth=*/1));
      // CUDA_SYNC_CHECK();

      // output_buffer.unmap();
    }

    //
    // Cleanup
    //
    {
      // CUDA_CHECK( cudaFree( reinterpret_cast<void*>( sbt.raygenRecord       )
      // ) ); CUDA_CHECK( cudaFree( reinterpret_cast<void*>( sbt.missRecordBase
      // ) ) ); CUDA_CHECK( cudaFree( reinterpret_cast<void*>(
      // sbt.hitgroupRecordBase ) ) );

      OPTIX_CHECK(optixPipelineDestroy(pipeline));
      OPTIX_CHECK(optixProgramGroupDestroy(hitgroup_prog_group));
      OPTIX_CHECK(optixProgramGroupDestroy(miss_prog_group));
      OPTIX_CHECK(optixProgramGroupDestroy(raygen_prog_group));
      OPTIX_CHECK(optixModuleDestroy(module));

      OPTIX_CHECK(optixDeviceContextDestroy(context));
    }
  }

  return 0;
}
