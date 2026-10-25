// arcinferzero NPU experiment (zero/npu/, separate from the engine): minimal declarations of Level Zero's NPU graph
// extension (ZE_extension_graph, Intel's level-zero-npu-extensions ze_graph_ext.h), written from its published layout
// and checked step by step against the installed driver (no copy of the header is on this machine). Only what the
// probes and benchmarks here use.
#pragma once
#include "../ze_min.h"

typedef struct _ze_graph_handle_t* ze_graph_handle_t;
typedef uint32_t ze_graph_format_t;
#define ZE_GRAPH_FORMAT_NATIVE 0x1u        // a compiled NPU blob
#define ZE_GRAPH_FORMAT_NGRAPH_LITE 0x2u   // a serialized OpenVINO IR (xml + weights): the driver's compiler builds it

#define ZE_STYPE_DEVICE_GRAPH_PROPERTIES 0x1001u
#define ZE_STYPE_GRAPH_DESC 0x1002u
#define ZE_STYPE_GRAPH_PROPERTIES 0x1003u
#define ZE_STYPE_GRAPH_ARGUMENT_PROPERTIES 0x1004u

struct ze_graph_compiler_version_info_t { uint16_t major, minor; };
struct ze_device_graph_properties_t {
  ze_structure_type_t stype;
  void* pNext;
  uint32_t graphExtensionVersion;
  ze_graph_compiler_version_info_t compilerVersion;
  ze_graph_format_t graphFormatsSupported;
  uint32_t maxOVOpsetVersionSupported;
};
struct ze_graph_desc_t {
  ze_structure_type_t stype;
  const void* pNext;
  ze_graph_format_t format;
  size_t inputSize;
  const uint8_t* pInput;
  const char* pBuildFlags;
};
struct ze_graph_properties_t {
  ze_structure_type_t stype;
  void* pNext;
  uint32_t numGraphArgs;
};
// argument types / precisions / layouts (ze_graph_argument_type_t, ze_graph_argument_precision_t, ...)
#define ZE_GRAPH_ARGUMENT_TYPE_INPUT 0x0u  // (as the driver reports them: 0 input, 1 output)
#define ZE_GRAPH_ARGUMENT_TYPE_OUTPUT 0x1u
#define ZE_MAX_GRAPH_ARGUMENT_NAME 256
#define ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE 5
struct ze_graph_argument_properties_t {
  ze_structure_type_t stype;
  void* pNext;
  char name[ZE_MAX_GRAPH_ARGUMENT_NAME];
  uint32_t type;
  uint32_t dims[ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE];
  uint32_t networkPrecision, networkLayout, devicePrecision, deviceLayout;
  // (newer drivers fill more fields behind these (quantization, debug names): room so they write into this struct)
  uint8_t reserved[1024];
};

typedef ze_result_t (*pfn_graph_create)(ze_context_handle_t, ze_device_handle_t, const ze_graph_desc_t*, ze_graph_handle_t*);
typedef ze_result_t (*pfn_graph_destroy)(ze_graph_handle_t);
typedef ze_result_t (*pfn_graph_get_properties)(ze_graph_handle_t, ze_graph_properties_t*);
typedef ze_result_t (*pfn_graph_get_argument_properties)(ze_graph_handle_t, uint32_t, ze_graph_argument_properties_t*);
typedef ze_result_t (*pfn_graph_set_argument_value)(ze_graph_handle_t, uint32_t, const void*);
typedef ze_result_t (*pfn_append_graph_initialize)(ze_command_list_handle_t, ze_graph_handle_t, ze_event_handle_t, uint32_t,
                                                   ze_event_handle_t*);
typedef ze_result_t (*pfn_append_graph_execute)(ze_command_list_handle_t, ze_graph_handle_t, void*, ze_event_handle_t,
                                                uint32_t, ze_event_handle_t*);
typedef ze_result_t (*pfn_graph_get_native_binary)(ze_graph_handle_t, size_t*, uint8_t*);
typedef ze_result_t (*pfn_device_get_graph_properties)(ze_device_handle_t, ze_device_graph_properties_t*);
typedef ze_result_t (*pfn_graph_build_log_get_string)(ze_graph_handle_t, uint32_t*, char*);

// the function table zeDriverGetExtensionFunctionAddress returns for "ZE_extension_graph_1_x" (fields in the
// header's order; the 1.1-1.3 entries this code does not call are kept as opaque pointers so the later ones line up)
struct ze_graph_dditable_t {
  pfn_graph_create pfnCreate;
  pfn_graph_destroy pfnDestroy;
  pfn_graph_get_properties pfnGetProperties;
  pfn_graph_get_argument_properties pfnGetArgumentProperties;
  pfn_graph_set_argument_value pfnSetArgumentValue;
  pfn_append_graph_initialize pfnAppendGraphInitialize;
  pfn_append_graph_execute pfnAppendGraphExecute;
  pfn_graph_get_native_binary pfnGetNativeBinary;
  pfn_device_get_graph_properties pfnDeviceGetGraphProperties;
  void* pfnGraphGetArgumentMetadata;        // 1.1
  void* pfnGetArgumentProperties2;          // 1.1
  void* pfnGetArgumentProperties3;          // 1.2
  void* pfnQueryNetworkCreate;              // 1.3
  void* pfnQueryNetworkDestroy;             // 1.3
  void* pfnQueryNetworkGetSupportedLayers;  // 1.3
  pfn_graph_build_log_get_string pfnBuildLogGetString;  // 1.4
};
