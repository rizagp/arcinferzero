// Vulkan against Level Zero on the same GPU, for what the API itself can change: (1) the memory read bandwidth a
// kernel reaches (membw_probe's GPU reader, same loads and grid) and (2) the cost of many small dependent kernels,
// as a generation step has (replay_probe's 700 kernels: re-recorded every step, and recorded once and replayed).
// The kernels' own speed is the shaders' (here the same simple ones), not the API's.
//   bin\vk_probe.exe [--gib 1] [--seconds 3] [--kernels 700] [--floats 65536] [--reps 30]
// Needs only the driver's vulkan-1.dll; the shaders (bench\vk\*.comp) are compiled to SPIR-V headers beforehand:
//   glslang -V --target-env vulkan1.2 --vn kRdSpv -o rd.spv.h rd.comp   (and step.comp as kStepSpv)
// Built with the Vulkan headers (Khronos Vulkan-Headers) on the include path.
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <windows.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include "vk/rd.spv.h"
#include "vk/step.spv.h"

#define VKF(X)                                                                                                     \
  X(vkEnumeratePhysicalDevices) X(vkGetPhysicalDeviceProperties) X(vkGetPhysicalDeviceQueueFamilyProperties)      \
  X(vkGetPhysicalDeviceMemoryProperties) X(vkCreateDevice) X(vkGetDeviceProcAddr)
#define VKD(X)                                                                                                     \
  X(vkGetDeviceQueue) X(vkCreateBuffer) X(vkGetBufferMemoryRequirements) X(vkAllocateMemory) X(vkBindBufferMemory) \
  X(vkCreateShaderModule) X(vkCreateDescriptorSetLayout) X(vkCreatePipelineLayout) X(vkCreateComputePipelines)    \
  X(vkCreateDescriptorPool) X(vkAllocateDescriptorSets) X(vkUpdateDescriptorSets) X(vkCreateCommandPool)          \
  X(vkAllocateCommandBuffers) X(vkBeginCommandBuffer) X(vkEndCommandBuffer) X(vkResetCommandBuffer)               \
  X(vkCmdBindPipeline) X(vkCmdBindDescriptorSets) X(vkCmdPushConstants) X(vkCmdDispatch) X(vkCmdPipelineBarrier)  \
  X(vkCmdFillBuffer) X(vkCreateFence) X(vkResetFences) X(vkWaitForFences) X(vkQueueSubmit) X(vkDeviceWaitIdle)
#define DECL(f) static PFN_##f f;
VKF(DECL)
VKD(DECL)
static PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr;
static PFN_vkCreateInstance vkCreateInstance;

#define CHECK(x)                                                                \
  do {                                                                          \
    const VkResult r_ = (x);                                                    \
    if (r_ != VK_SUCCESS) { fprintf(stderr, "%s: VkResult %d\n", #x, (int)r_); exit(1); } \
  } while (0)

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t) { return std::chrono::duration<double, std::milli>(clk::now() - t).count(); }
static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

int main(int argc, char** argv) {
  double gib = 1.0, seconds = 3.0;
  int nk = 700, nf = 65536, reps = 30;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string a = argv[i];
    if (a == "--gib") gib = atof(argv[i + 1]);
    else if (a == "--seconds") seconds = atof(argv[i + 1]);
    else if (a == "--kernels") nk = atoi(argv[i + 1]);
    else if (a == "--floats") nf = atoi(argv[i + 1]);
    else if (a == "--reps") reps = atoi(argv[i + 1]);
  }
  HMODULE lib = LoadLibraryA("vulkan-1.dll");
  if (!lib) { fprintf(stderr, "no vulkan-1.dll\n"); return 1; }
  vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)GetProcAddress(lib, "vkGetInstanceProcAddr");
  vkCreateInstance = (PFN_vkCreateInstance)vkGetInstanceProcAddr(nullptr, "vkCreateInstance");
  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.pApplicationName = "vk_probe";
  app.apiVersion = VK_API_VERSION_1_2;
  VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ici.pApplicationInfo = &app;
  VkInstance inst;
  CHECK(vkCreateInstance(&ici, nullptr, &inst));
#define LOADI(f) f = (PFN_##f)vkGetInstanceProcAddr(inst, #f);
  VKF(LOADI)

  // the GPU: an integrated one first (the laptop's), else the first
  uint32_t np = 0;
  vkEnumeratePhysicalDevices(inst, &np, nullptr);
  std::vector<VkPhysicalDevice> pds(np);
  vkEnumeratePhysicalDevices(inst, &np, pds.data());
  if (!np) { fprintf(stderr, "no Vulkan device\n"); return 1; }
  VkPhysicalDevice pd = pds[0];
  for (VkPhysicalDevice p : pds) {
    VkPhysicalDeviceProperties pr;
    vkGetPhysicalDeviceProperties(p, &pr);
    if (pr.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) { pd = p; break; }
  }
  VkPhysicalDeviceProperties props;
  vkGetPhysicalDeviceProperties(pd, &props);
  uint32_t nq = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, nullptr);
  std::vector<VkQueueFamilyProperties> qf(nq);
  vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qf.data());
  uint32_t fam = UINT32_MAX;
  for (uint32_t i = 0; i < nq && fam == UINT32_MAX; ++i)
    if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) fam = i;
  const float prio = 1.0f;
  VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qci.queueFamilyIndex = fam;
  qci.queueCount = 1;
  qci.pQueuePriorities = &prio;
  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qci;
  VkDevice dev;
  CHECK(vkCreateDevice(pd, &dci, nullptr, &dev));
#define LOADD(f) f = (PFN_##f)vkGetDeviceProcAddr(dev, #f);
  VKD(LOADD)
  VkQueue q;
  vkGetDeviceQueue(dev, fam, 0, &q);
  printf("%s, Vulkan %u.%u.%u, driver %u\n", props.deviceName, VK_API_VERSION_MAJOR(props.apiVersion),
         VK_API_VERSION_MINOR(props.apiVersion), VK_API_VERSION_PATCH(props.apiVersion), props.driverVersion);

  VkPhysicalDeviceMemoryProperties mp;
  vkGetPhysicalDeviceMemoryProperties(pd, &mp);
  auto buffer = [&](VkDeviceSize size, VkBuffer& b) {
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = size;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    CHECK(vkCreateBuffer(dev, &bci, nullptr, &b));
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(dev, b, &mr);
    uint32_t type = UINT32_MAX;
    for (uint32_t i = 0; i < mp.memoryTypeCount && type == UINT32_MAX; ++i)
      if ((mr.memoryTypeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) type = i;
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = type;
    VkDeviceMemory m;
    CHECK(vkAllocateMemory(dev, &mai, nullptr, &m));
    CHECK(vkBindBufferMemory(dev, b, m, 0));
  };
  // one layout for both shaders: two storage buffers and an int push constant
  VkDescriptorSetLayoutBinding bind[2] = {};
  for (int i = 0; i < 2; ++i) {
    bind[i].binding = i;
    bind[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bind[i].descriptorCount = 1;
    bind[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }
  VkDescriptorSetLayoutCreateInfo dl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  dl.bindingCount = 2;
  dl.pBindings = bind;
  VkDescriptorSetLayout dsl;
  CHECK(vkCreateDescriptorSetLayout(dev, &dl, nullptr, &dsl));
  VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 4};
  VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  plci.setLayoutCount = 1;
  plci.pSetLayouts = &dsl;
  plci.pushConstantRangeCount = 1;
  plci.pPushConstantRanges = &pcr;
  VkPipelineLayout pl;
  CHECK(vkCreatePipelineLayout(dev, &plci, nullptr, &pl));
  auto pipeline = [&](const uint32_t* code, size_t bytes) {
    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = bytes;
    smci.pCode = code;
    VkShaderModule sm;
    CHECK(vkCreateShaderModule(dev, &smci, nullptr, &sm));
    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = sm;
    cpci.stage.pName = "main";
    cpci.layout = pl;
    VkPipeline p;
    CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &p));
    return p;
  };
  VkPipeline prd = pipeline(kRdSpv, sizeof kRdSpv), pstep = pipeline(kStepSpv, sizeof kStepSpv);
  VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4};
  VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  dpci.maxSets = 2;
  dpci.poolSizeCount = 1;
  dpci.pPoolSizes = &ps;
  VkDescriptorPool dp;
  CHECK(vkCreateDescriptorPool(dev, &dpci, nullptr, &dp));
  auto set = [&](VkBuffer a, VkBuffer b) {
    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = dp;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &dsl;
    VkDescriptorSet s;
    CHECK(vkAllocateDescriptorSets(dev, &dsai, &s));
    VkDescriptorBufferInfo bi[2] = {{a, 0, VK_WHOLE_SIZE}, {b, 0, VK_WHOLE_SIZE}};
    VkWriteDescriptorSet w[2] = {};
    for (int i = 0; i < 2; ++i) {
      w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      w[i].dstSet = s;
      w[i].dstBinding = i;
      w[i].descriptorCount = 1;
      w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      w[i].pBufferInfo = &bi[i];
    }
    vkUpdateDescriptorSets(dev, 2, w, 0, nullptr);
    return s;
  };
  VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  cpi.queueFamilyIndex = fam;
  VkCommandPool pool;
  CHECK(vkCreateCommandPool(dev, &cpi, nullptr, &pool));
  VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cbai.commandPool = pool;
  cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbai.commandBufferCount = 1;
  VkCommandBuffer cb;
  CHECK(vkAllocateCommandBuffers(dev, &cbai, &cb));
  VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VkFence fence;
  CHECK(vkCreateFence(dev, &fci, nullptr, &fence));
  auto submit_wait = [&] {
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    CHECK(vkResetFences(dev, 1, &fence));
    CHECK(vkQueueSubmit(q, 1, &si, fence));
    CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));
  };
  VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  auto barrier = [&] {
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
  };

  // 1. read bandwidth: 1 GiB read over and over, 2048 work-groups of 256 (membw_probe's default)
  {
    const VkDeviceSize bytes = ((VkDeviceSize)(gib * (1ull << 30)) + 4095) / 4096 * 4096;
    VkBuffer x, o;
    buffer(bytes, x);
    buffer(64, o);
    VkDescriptorSet s = set(x, o);
    CHECK(vkBeginCommandBuffer(cb, &cbbi));
    vkCmdFillBuffer(cb, x, 0, VK_WHOLE_SIZE, 0x01010101u);
    vkCmdFillBuffer(cb, o, 0, VK_WHOLE_SIZE, 0u);
    CHECK(vkEndCommandBuffer(cb));
    submit_wait();
    const int n4 = (int)(bytes / 16);
    CHECK(vkResetCommandBuffer(cb, 0));
    CHECK(vkBeginCommandBuffer(cb, &cbbi));
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, prd);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &s, 0, nullptr);
    vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &n4);
    vkCmdDispatch(cb, 2048, 1, 1);
    CHECK(vkEndCommandBuffer(cb));
    submit_wait();  // (first touch)
    uint64_t done = 0;
    const auto t0 = clk::now();
    while (ms_since(t0) < seconds * 1000) { submit_wait(); done += bytes; }
    printf("read bandwidth: %.1f GB/s (%.2f GiB buffer, %.1f s)\n", done / (ms_since(t0) / 1000) / 1e9, bytes / 1073741824.0, seconds);
  }

  // 2. nk small dependent kernels per step: re-recorded every step (as the engine appends to its list now), and
  //    recorded once and replayed
  {
    VkBuffer y, x;
    buffer((VkDeviceSize)nf * 4, y);
    buffer((VkDeviceSize)nf * 4, x);
    VkDescriptorSet s = set(y, x);
    const uint32_t groups = (uint32_t)((nf + 255) / 256);
    auto record = [&] {
      CHECK(vkBeginCommandBuffer(cb, &cbbi));
      vkCmdFillBuffer(cb, y, 0, VK_WHOLE_SIZE, 0u);
      vkCmdFillBuffer(cb, x, 0, VK_WHOLE_SIZE, 0u);
      barrier();
      vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pstep);
      vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &s, 0, nullptr);
      vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &nf);
      for (int i = 0; i < nk; ++i) {
        vkCmdDispatch(cb, groups, 1, 1);
        barrier();
      }
      CHECK(vkEndCommandBuffer(cb));
    };
    std::vector<double> rerec, replay;
    for (int r = 0; r < reps; ++r) {
      const auto t = clk::now();
      CHECK(vkResetCommandBuffer(cb, 0));
      record();
      submit_wait();
      rerec.push_back(ms_since(t));
    }
    CHECK(vkResetCommandBuffer(cb, 0));
    const auto tr = clk::now();
    record();
    const double rec_ms = ms_since(tr);
    for (int r = 0; r < reps; ++r) {
      const auto t = clk::now();
      submit_wait();
      replay.push_back(ms_since(t));
    }
    printf("%d dependent kernels of %d floats each, median of %d steps\n", nk, nf, reps);
    printf("%-40s %8.3f ms per step (%.2f us per kernel)\n", "recorded every step", median(rerec), median(rerec) * 1000 / nk);
    printf("%-40s %8.3f ms per step (%.2f us per kernel; recorded once in %.2f ms)\n", "recorded once, replayed",
           median(replay), median(replay) * 1000 / nk, rec_ms);
  }
  vkDeviceWaitIdle(dev);
  return 0;
}
