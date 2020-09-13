//
// Based on optixHello sample in OptiX SDK
//
#include <array>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

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

#define CU_SYNC_CHECK()                                                   \
  do {                                                                    \
    /* assume cuCtxSynchronize() ~= cudaDeviceSynchronize() */            \
    CUresult ret = cuCtxSynchronize();                                    \
    if (ret != CUDA_SUCCESS) {                                            \
      std::cerr << __FILE__ << ":" << __LINE__                            \
                << " cuCtxSynchronize() failed. retcode " << ret << "\n"; \
      exit(-1);                                                           \
    }                                                                     \
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

// struct Params {
//  CUdeviceptr image;  // RGBA
//  unsigned int image_width;
//};

// assume at least 4byte aligned
struct float3 {
  float x, y, z;
};

struct RayGenData {
  // no data
};

struct MissData {
  float3 bg_color;
};

struct HitGroupData {
  // no data
};

struct Params {
  CUdeviceptr image;  // uchar4*
  unsigned int image_width;
  unsigned int image_height;
  float3 cam_eye;
  float3 cam_u, cam_v, cam_w;
  OptixTraversableHandle handle;
};

template <typename T>
struct SbtRecord {
#ifdef _MSC_VER
  __declspec(align(
      OPTIX_SBT_RECORD_ALIGNMENT)) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
#else  // assume gcc or clang
  char header[OPTIX_SBT_RECORD_HEADER_SIZE]
      __attribute__((aligned(OPTIX_SBT_RECORD_ALIGNMENT)));
#endif
  T data;
};

typedef SbtRecord<RayGenData> RayGenSbtRecord;
typedef SbtRecord<MissData> MissSbtRecord;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;

bool CUDAAllocDeviceMem(CUdeviceptr* dptr, size_t sz) {
  CU_CHECK(cuMemAlloc(dptr, sz));

  return true;
}

static void context_log_cb(unsigned int level, const char* tag,
                           const char* message, void* /*cbdata */) {
  std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag
            << "]: " << message << "\n";
}

bool LoadPTXFromFile(const std::string& filename, std::string* output) {
  std::ifstream ifs(filename);
  if (!ifs) {
    return false;
  }

  std::stringstream ss;
  ss << ifs.rdbuf();

  (*output) = ss.str();

  return true;
}

static void BuildAccel(const OptixDeviceContext& context,
                       OptixTraversableHandle* handle_out,
                       CUdeviceptr* ptr_out) {
  //
  // accel handling
  //
  OptixTraversableHandle gas_handle;
  CUdeviceptr d_gas_output_buffer;
  {
    // Use default options for simplicity.  In a real use case we would want to
    // enable compaction, etc
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    // Triangle build input: simple list of three vertices
    const std::array<float3, 3> vertices = {
        {{-0.5f, -0.5f, 0.0f}, {0.5f, -0.5f, 0.0f}, {0.0f, 0.5f, 0.0f}}};

    const size_t vertices_size = sizeof(float3) * vertices.size();
    CUdeviceptr d_vertices = 0;
    CUDAAllocDeviceMem(&d_vertices, vertices_size);

    // CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_vertices ),
    // vertices_size ) );

    std::cout << "vertices_size: " << vertices_size << "\n";

    CU_CHECK(cuMemcpyHtoD(d_vertices, vertices.data(), vertices_size));
    // CUDA_CHECK( cudaMemcpy(
    //            reinterpret_cast<void*>( d_vertices ),
    //            vertices.data(),
    //            vertices_size,
    //            cudaMemcpyHostToDevice
    //            ) );

    // Our build input is a simple list of non-indexed triangle vertices
    const uint32_t triangle_input_flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
    OptixBuildInput triangle_input = {};
    triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangle_input.triangleArray.numVertices =
        static_cast<uint32_t>(vertices.size());
    triangle_input.triangleArray.vertexBuffers = &d_vertices;
    triangle_input.triangleArray.flags = triangle_input_flags;
    triangle_input.triangleArray.numSbtRecords = 1;

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &accel_options,
                                             &triangle_input,
                                             1,  // Number of build inputs
                                             &gas_buffer_sizes));
    CUdeviceptr d_temp_buffer_gas;

    CUDAAllocDeviceMem(&d_temp_buffer_gas, gas_buffer_sizes.tempSizeInBytes);
    CUDAAllocDeviceMem(&d_gas_output_buffer,
                       gas_buffer_sizes.outputSizeInBytes);

    // CUDA_CHECK( cudaMalloc(
    //            reinterpret_cast<void**>( &d_temp_buffer_gas ),
    //            gas_buffer_sizes.tempSizeInBytes
    //            ) );
    // CUDA_CHECK( cudaMalloc(
    //            reinterpret_cast<void**>( &d_gas_output_buffer ),
    //            gas_buffer_sizes.outputSizeInBytes
    //            ) );

    OPTIX_CHECK(optixAccelBuild(
        context,
        0,  // CUDA stream
        &accel_options, &triangle_input,
        1,  // num build inputs
        d_temp_buffer_gas, gas_buffer_sizes.tempSizeInBytes,
        d_gas_output_buffer, gas_buffer_sizes.outputSizeInBytes, &gas_handle,
        nullptr,  // emitted property list
        0         // num emitted properties
        ));

    // We can now free the scratch space buffer used during build and the vertex
    // inputs, since they are not needed by our trivial shading method
    // CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_temp_buffer_gas ) ) );
    // CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_vertices        ) ) );

    CU_CHECK(cuMemFree(d_temp_buffer_gas));
    CU_CHECK(cuMemFree(d_vertices));
  }

  std::cout << "gas_handle: " << gas_handle << "\n";

  (*handle_out) = gas_handle;
  (*ptr_out) = d_gas_output_buffer;
}

int main(int argc, char** argv) {
  const std::string ptx_filename = "../data/optixTriangle.ptx";

  static_assert(
      offsetof(RayGenSbtRecord, data) % OPTIX_SBT_RECORD_ALIGNMENT == 0,
      "Member variable must be aligned to multiple of OPTIX_SBT_RECORD_ALIGNMENT(=16)");

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
  }

  //
  // Initialize OptiX and launch kernel.
  //

  {
    if (optixInit() != OPTIX_SUCCESS) {
      std::cerr << "Failed to initialize OptiX\n";
      return -1;
    }

    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &context_log_cb;
    options.logCallbackLevel = 4;
    OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));

    //
    // BVH setup
    //

    OptixTraversableHandle gas_handle;
    CUdeviceptr d_gas_output_buffer;
    BuildAccel(context, &gas_handle, &d_gas_output_buffer);


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
          OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
      pipeline_compile_options.numPayloadValues = 3;
      pipeline_compile_options.numAttributeValues = 3;

#if 1  // DEBUG  // Enables debug exceptions during optix launches. This may
       // incur significant performance cost and should only be done during
       // development.
      pipeline_compile_options.exceptionFlags =
          OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
          OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
#else
      pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif

      pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
      pipeline_compile_options.usesPrimitiveTypeFlags =
          OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

      std::string ptx;
      if (!LoadPTXFromFile(ptx_filename, &ptx)) {
        std::cerr << "Failed to open/read PTX file: " << ptx_filename << "\n";
        exit(-1);
      }
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
      raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
      size_t sizeof_log = sizeof(logbuf);
      OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &raygen_prog_group_desc,
                                              1,  // num program groups
                                              &program_group_options, logbuf,
                                              &sizeof_log, &raygen_prog_group));

      // Leave miss group's module and entryfunc name null
      OptixProgramGroupDesc miss_prog_group_desc = {};
      miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
      miss_prog_group_desc.miss.module = module;
      miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
      sizeof_log = sizeof(logbuf);
      OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &miss_prog_group_desc,
                                              1,  // num program groups
                                              &program_group_options, logbuf,
                                              &sizeof_log, &miss_prog_group));

      // Leave hit group's module and entryfunc name null
      OptixProgramGroupDesc hitgroup_prog_group_desc = {};
      hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
      hitgroup_prog_group_desc.hitgroup.moduleCH = module;
      hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH =
          "__closesthit__ch";
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
      const uint32_t max_trace_depth = 1;
      OptixProgramGroup program_groups[] = {raygen_prog_group, miss_prog_group,
                                            hitgroup_prog_group};

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

      std::cout << "trav: " << direct_callable_stack_size_from_traversal
                << ", state: " << direct_callable_stack_size_from_state << "\n";

      OPTIX_CHECK(optixPipelineSetStackSize(
          pipeline, direct_callable_stack_size_from_traversal,
          direct_callable_stack_size_from_state, continuation_stack_size,
          1  // maxTraversableDepth(must be 1 for single gas graphs)
          ));
    }

    //
    // Set up shader binding table
    //
    OptixShaderBindingTable sbt = {};
    {
      CUdeviceptr raygen_record;
      const size_t raygen_record_size = sizeof(RayGenSbtRecord);
      CUDAAllocDeviceMem(&raygen_record, raygen_record_size);
      RayGenSbtRecord rg_sbt;
      OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group, &rg_sbt));
      // rg_sbt.data = {0.462f, 0.725f, 0.f};

      CU_CHECK(cuMemcpyHtoD(raygen_record, &rg_sbt, raygen_record_size));

      CUdeviceptr miss_record;
      size_t miss_record_size = sizeof(MissSbtRecord);
      CUDAAllocDeviceMem(&miss_record, miss_record_size);
      MissSbtRecord ms_sbt;
      ms_sbt.data = {0.3f, 0.1f, 0.2f};
      OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_group, &ms_sbt));

      CU_CHECK(cuMemcpyHtoD(miss_record, &ms_sbt, miss_record_size));

      CUdeviceptr hitgroup_record;
      size_t hitgroup_record_size = sizeof(HitGroupSbtRecord);
      CUDAAllocDeviceMem(&hitgroup_record, hitgroup_record_size);
      HitGroupSbtRecord hg_sbt;
      OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_group, &hg_sbt));
      CU_CHECK(cuMemcpyHtoD(hitgroup_record, &hg_sbt, hitgroup_record_size));

      sbt.raygenRecord = raygen_record;
      sbt.missRecordBase = miss_record;
      sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
      sbt.missRecordCount = 1;
      sbt.hitgroupRecordBase = hitgroup_record;
      sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
      sbt.hitgroupRecordCount = 1;
      // std::cout << "hitGroupSz " << sizeof(HitGroupSbtRecord) << "\n";
    }

    int width = 1024;
    int height = 768;
    CUdeviceptr d_image;  // size = uchar4 * width * height
    CUDAAllocDeviceMem(&d_image, 4 * width * height);

    //
    // launch
    //
    Params params;
    {
      CUstream stream;
      CU_CHECK(cuStreamCreate(&stream, CU_STREAM_DEFAULT));

      params.image = d_image;
      params.image_width = width;
      params.image_height = height;
      params.handle = gas_handle;

      std::cout << "offsetof(image_width) = " << offsetof(Params, image_width) << "\n";
      std::cout << "offsetof(image_height) = " << offsetof(Params, image_height) << "\n";
      std::cout << "offsetof(cam_u) = " << offsetof(Params, cam_u) << "\n";
      std::cout << "offsetof(cam_v) = " << offsetof(Params, cam_v) << "\n";
      std::cout << "offsetof(cam_w) = " << offsetof(Params, cam_w) << "\n";
      std::cout << "offsetof(handle) = " << offsetof(Params, handle) << "\n";

      // hardcoded camera config.
      params.cam_eye.x = 0.0f;
      params.cam_eye.y = 0.0f;
      params.cam_eye.z = 2.0f;

      // Values are calculated from 1024x768
      params.cam_u.x = 1.10457f;
      params.cam_u.y = -0.0f;
      params.cam_u.z = 0.0f;

      params.cam_v.x = 0.0f;
      params.cam_v.y = 0.828427f;
      params.cam_v.z = 0.0f;

      params.cam_w.x = 0.0f;
      params.cam_w.y = 0.0f;
      params.cam_w.z = -2.0f;

      CUdeviceptr d_param;
      CUDAAllocDeviceMem(&d_param, sizeof(Params));
      CU_CHECK(cuMemcpyHtoD(d_param, &params, sizeof(params)));

      OPTIX_CHECK(optixLaunch(pipeline, stream, d_param, sizeof(Params), &sbt,
                              width, height, /*depth=*/1));

      CU_SYNC_CHECK();

      std::cout << "Launch OK!\n";
    }

    //
    // Readback result.
    //
    std::vector<uint8_t> h_image;
    h_image.resize(4 * width * height, 0);

    CU_CHECK(cuMemcpyDtoH(reinterpret_cast<void*>(h_image.data()), params.image,
                          h_image.size()));

    //
    // Save result to a file.
    //
    {
      int n = stbi_write_png("output.png", width, height, /* comp */ 4,
                             h_image.data(), /* stride */ 0);
      if (n < 1) {
        std::cerr << "Failed to write PNG image.\n";
        exit(-1);
      }
    }

    //
    // Cleanup
    //
    {
      CU_CHECK(cuMemFree(sbt.raygenRecord));
      CU_CHECK(cuMemFree(sbt.missRecordBase));
      CU_CHECK(cuMemFree(sbt.hitgroupRecordBase));
      CU_CHECK(cuMemFree(d_gas_output_buffer));

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
