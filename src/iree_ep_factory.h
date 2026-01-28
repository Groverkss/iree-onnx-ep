// iree_ep_factory.h - IREE Execution Provider Factory

#ifndef IREE_ONNX_EP_SRC_IREE_EP_FACTORY_H_
#define IREE_ONNX_EP_SRC_IREE_EP_FACTORY_H_

#include <string>
#include <vector>

#include "iree_wrappers.h"
#include "ort_import.h"

namespace iree_onnx_ep {

// EP configuration constants
inline constexpr const char* kEpVendor = "IREE";
inline constexpr uint32_t kEpVendorId = 0x1EEE;  // "IREE" in hex-ish
inline constexpr const char* kEpVersion = "0.1.0";

// Hardware vendor IDs for device matching.
// These match OrtDevice::VendorIds from onnxruntime/core/framework/ortdevice.h.
namespace VendorIds {
inline constexpr uint32_t kAmd = 0x1002;     // AMD: ROCm, MIGraphX EPs
inline constexpr uint32_t kNvidia = 0x10DE;  // NVIDIA: CUDA/TensorRT
inline constexpr uint32_t kIntel = 0x8086;   // Intel: OpenVINO
}  // namespace VendorIds

// Helper struct to pass API pointers
struct ApiPtrs {
  const OrtApi& ort_api;
  const OrtEpApi& ep_api;
  const OrtModelEditorApi& model_editor_api;
};

// IREE Execution Provider Factory
// Inherits from OrtEpFactory and sets up function pointers in constructor.
// Manages a shared IREE runtime instance that is used by all EPs created
// by this factory.
class IreeEpFactory : public OrtEpFactory, public ApiPtrs {
 public:
  IreeEpFactory(const char* ep_name, ApiPtrs apis,
                const OrtLogger* default_logger);
  ~IreeEpFactory();

  // Accessor for the shared IREE runtime instance.
  iree_runtime_instance_t* IreeInstance() const { return instance_.Get(); }

 private:
  // Factory interface implementations (called via function pointers)
  static const char* ORT_API_CALL
  GetNameImpl(const OrtEpFactory* this_ptr) noexcept;
  static const char* ORT_API_CALL
  GetVendorImpl(const OrtEpFactory* this_ptr) noexcept;
  static uint32_t ORT_API_CALL
  GetVendorIdImpl(const OrtEpFactory* this_ptr) noexcept;
  static const char* ORT_API_CALL
  GetVersionImpl(const OrtEpFactory* this_ptr) noexcept;

  static OrtStatus* ORT_API_CALL GetSupportedDevicesImpl(
      OrtEpFactory* this_ptr, const OrtHardwareDevice* const* devices,
      size_t num_devices, OrtEpDevice** ep_devices, size_t max_ep_devices,
      size_t* p_num_ep_devices) noexcept;

  static OrtStatus* ORT_API_CALL
  CreateEpImpl(OrtEpFactory* this_ptr, const OrtHardwareDevice* const* devices,
               const OrtKeyValuePairs* const* ep_metadata, size_t num_devices,
               const OrtSessionOptions* session_options,
               const OrtLogger* logger, OrtEp** ep) noexcept;

  static void ORT_API_CALL ReleaseEpImpl(OrtEpFactory* this_ptr,
                                         OrtEp* ep) noexcept;

  static OrtStatus* ORT_API_CALL
  CreateAllocatorImpl(OrtEpFactory* this_ptr, const OrtMemoryInfo* memory_info,
                      const OrtKeyValuePairs* allocator_options,
                      OrtAllocator** allocator) noexcept;

  static void ORT_API_CALL ReleaseAllocatorImpl(
      OrtEpFactory* this_ptr, OrtAllocator* allocator) noexcept;

  static OrtStatus* ORT_API_CALL CreateDataTransferImpl(
      OrtEpFactory* this_ptr, OrtDataTransferImpl** data_transfer) noexcept;

  static bool ORT_API_CALL
  IsStreamAwareImpl(const OrtEpFactory* this_ptr) noexcept;

  static OrtStatus* ORT_API_CALL CreateSyncStreamForDeviceImpl(
      OrtEpFactory* this_ptr, const OrtMemoryDevice* memory_device,
      const OrtKeyValuePairs* stream_options,
      OrtSyncStreamImpl** stream) noexcept;

  // Enumerate IREE devices and create OrtHardwareDevice instances.
  void CreateIreeHwDevices();

  // Member variables
  Ort::Logger logger_;
  const std::string ep_name_;

  // IREE runtime instance (shared across all EPs created by this factory).
  RuntimeInstancePtr instance_;

  // OrtHardwareDevice instances created by this factory.
  // Owned by the factory and released in the destructor.
  std::vector<OrtHardwareDevice*> hw_devices_;
};

}  // namespace iree_onnx_ep

#endif  // IREE_ONNX_EP_SRC_IREE_EP_FACTORY_H_
