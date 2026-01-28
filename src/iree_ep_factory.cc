// iree_ep_factory.cc - IREE Execution Provider Factory Implementation

#include "iree_ep_factory.h"

#include <memory>

#include "iree_ep.h"

namespace iree_onnx_ep {

IreeEpFactory::IreeEpFactory(const char* ep_name, ApiPtrs apis,
                             const OrtLogger* default_logger)
    : OrtEpFactory{},
      ApiPtrs(apis),
      logger_(default_logger),
      ep_name_(ep_name) {
  // Set ORT version we support.
  ort_version_supported = ORT_API_VERSION;

  // Set function pointers.
  GetName = GetNameImpl;
  GetVendor = GetVendorImpl;
  GetVendorId = GetVendorIdImpl;
  GetVersion = GetVersionImpl;
  GetSupportedDevices = GetSupportedDevicesImpl;
  CreateEp = CreateEpImpl;
  ReleaseEp = ReleaseEpImpl;
  CreateAllocator = CreateAllocatorImpl;
  ReleaseAllocator = ReleaseAllocatorImpl;
  CreateDataTransfer = CreateDataTransferImpl;
  IsStreamAware = IsStreamAwareImpl;
  CreateSyncStreamForDevice = CreateSyncStreamForDeviceImpl;

  // Initialize IREE runtime instance (shared across all EPs).
  iree_runtime_instance_options_t instance_options;
  iree_runtime_instance_options_initialize(&instance_options);
  iree_runtime_instance_options_use_all_available_drivers(&instance_options);

  iree_status_t status = iree_runtime_instance_create(
      &instance_options, iree_allocator_system(), instance_.ForOutput());
  if (!iree_status_is_ok(status)) {
    // Log error but continue - EP creation will fail later if needed.
    ORT_CXX_LOG_NOEXCEPT(logger_, ORT_LOGGING_LEVEL_ERROR,
                         "IREE EP: Failed to create IREE runtime instance");
    iree_status_ignore(status);
    return;
  }

  CreateIreeHwDevices();
}

void IreeEpFactory::CreateIreeHwDevices() {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_driver_registry_t* registry =
      iree_runtime_instance_driver_registry(instance_.Get());

  iree_host_size_t driver_count = 0;
  IreeAllocatedPtr<iree_hal_driver_info_t> driver_infos(allocator);
  iree_status_t status = iree_hal_driver_registry_enumerate(
      registry, allocator, &driver_count, driver_infos.ForOutput());
  if (!iree_status_is_ok(status)) {
    ORT_CXX_LOG_NOEXCEPT(logger_, ORT_LOGGING_LEVEL_WARNING,
                         "IREE EP: Failed to enumerate drivers");
    iree_status_ignore(status);
    return;
  }

  size_t device_index = 0;
  for (iree_host_size_t i = 0; i < driver_count; ++i) {
    iree_string_view_t driver_name = driver_infos.Get()[i].driver_name;
    std::string driver_name_str(driver_name.data, driver_name.size);

    HalDriverPtr driver;
    status = iree_hal_driver_registry_try_create(registry, driver_name,
                                                 allocator, driver.ForOutput());
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      continue;
    }

    iree_host_size_t device_count = 0;
    IreeAllocatedPtr<iree_hal_device_info_t> device_infos(allocator);
    status = iree_hal_driver_query_available_devices(
        driver.Get(), allocator, &device_count, device_infos.ForOutput());
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      continue;
    }

    // Create OrtHardwareDevice for each IREE device available.
    //
    // Given an OrtHardwareDevice, ORT can find which hardware device it is by
    // checking its vendor id (should be kEpVendorId) and the device id (the
    // index it's stored in hw_devices_).
    //
    // IREE can identify the device using the driver and device path stored in
    // the hardware device metadata.
    for (iree_host_size_t j = 0; j < device_count; ++j) {
      iree_string_view_t device_path = device_infos.Get()[j].path;
      std::string device_path_str(device_path.data, device_path.size);

      OrtKeyValuePairs* hw_metadata = nullptr;
      ort_api.CreateKeyValuePairs(&hw_metadata);
      ort_api.AddKeyValuePair(hw_metadata, "iree.driver",
                              driver_name_str.c_str());
      ort_api.AddKeyValuePair(hw_metadata, "iree.device_path",
                              device_path_str.c_str());

      // TODO: We are pretending here that all IREE devices are GPUs. This is
      // because we follow a host <-> device model in IREE. How does this
      // matter in practice? If we are using IREE APIs anyway, does it matter
      // if we set this to CPU?
      OrtHardwareDeviceType device_type = OrtHardwareDeviceType_GPU;

      OrtHardwareDevice* hw_device = nullptr;
      OrtStatus* ort_status = ep_api.CreateHardwareDevice(
          device_type, kEpVendorId, static_cast<uint32_t>(device_index++),
          kEpVendor, hw_metadata, &hw_device);
      ort_api.ReleaseKeyValuePairs(hw_metadata);
      if (ort_status != nullptr) {
        ort_api.ReleaseStatus(ort_status);
        continue;
      }

      hw_devices_.push_back(hw_device);
    }
  }
}

IreeEpFactory::~IreeEpFactory() {
  for (auto* hw_device : hw_devices_) {
    ep_api.ReleaseHardwareDevice(hw_device);
  }
}

// Factory interface implementations

/*static*/
const char* ORT_API_CALL
IreeEpFactory::GetNameImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto* factory = static_cast<const IreeEpFactory*>(this_ptr);
  return factory->ep_name_.c_str();
}

/*static*/
const char* ORT_API_CALL
IreeEpFactory::GetVendorImpl(const OrtEpFactory* /*this_ptr*/) noexcept {
  return kEpVendor;
}

/*static*/
uint32_t ORT_API_CALL
IreeEpFactory::GetVendorIdImpl(const OrtEpFactory* /*this_ptr*/) noexcept {
  return kEpVendorId;
}

/*static*/
const char* ORT_API_CALL
IreeEpFactory::GetVersionImpl(const OrtEpFactory* /*this_ptr*/) noexcept {
  return kEpVersion;
}

/*static*/
OrtStatus* ORT_API_CALL IreeEpFactory::GetSupportedDevicesImpl(
    OrtEpFactory* this_ptr, const OrtHardwareDevice* const* /*devices*/,
    size_t /*num_devices*/, OrtEpDevice** ep_devices, size_t max_ep_devices,
    size_t* p_num_ep_devices) noexcept {
  auto* factory = static_cast<IreeEpFactory*>(this_ptr);
  size_t& num_ep_devices = *p_num_ep_devices;
  num_ep_devices = 0;

  // Create OrtEpDevice for each hardware device enumerated in constructor.
  for (size_t i = 0;
       i < factory->hw_devices_.size() && num_ep_devices < max_ep_devices;
       ++i) {
    OrtEpDevice* ep_device = nullptr;
    ORT_RETURN_IF_ERROR(factory->ep_api.CreateEpDevice(
        factory, factory->hw_devices_[i], nullptr, nullptr, &ep_device));
    ep_devices[num_ep_devices++] = ep_device;
  }

  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL IreeEpFactory::CreateEpImpl(
    OrtEpFactory* this_ptr, const OrtHardwareDevice* const* devices,
    const OrtKeyValuePairs* const* /*ep_options*/, size_t num_devices,
    const OrtSessionOptions* session_options, const OrtLogger* logger,
    OrtEp** ep) noexcept {
  auto* factory = static_cast<IreeEpFactory*>(this_ptr);
  *ep = nullptr;

  // Validate device selection.
  if (num_devices != 1) {
    // TODO: Can we even do multi-device work with onnxruntime? Afaik, there is
    // multi-device graph partitioning support:
    // https://github.com/microsoft/onnxruntime/issues/16382
    //
    // The other option would be to partition the graph ourselves in IREE. That
    // seems like a whole other line of work which is very out-of-scope for now.
    return Ort::Status("NYI: Multi-device IREE EP", ORT_NOT_IMPLEMENTED)
        .release();
  }

  // Get driver and device path from hardware device metadata.
  const OrtHardwareDevice& hardware_device = *devices[0];
  const OrtKeyValuePairs* hw_metadata =
      factory->ort_api.HardwareDevice_Metadata(&hardware_device);
  const char* driver_name =
      factory->ort_api.GetKeyValue(hw_metadata, "iree.driver");
  const char* device_path =
      factory->ort_api.GetKeyValue(hw_metadata, "iree.device_path");
  if (driver_name == nullptr || device_path == nullptr) {
    return Ort::Status(
               "IREE EP: Driver or device path not found in hardware device "
               "metadata",
               ORT_INVALID_ARGUMENT)
        .release();
  }

  // Parse configuration from session options config entries.
  // The Python API stores options with "ep.iree." prefix.
  IreeEp::Config config = {};
  if (session_options != nullptr) {
    Ort::ConstSessionOptions sess_opts(session_options);
    config.target_arch =
        sess_opts.GetConfigEntryOrDefault("ep.iree.target_arch", "");
    config.opt_level =
        sess_opts.GetConfigEntryOrDefault("ep.iree.opt_level", "O0");
    config.save_intermediates = sess_opts.GetConfigEntryOrDefault(
                                    "ep.iree.save_intermediates", "0") == "1";
  }

  // Select backend based on driver.
  std::string driver_str(driver_name);
  if (driver_str == "vulkan") {
    config.backend = "vulkan";
  } else if (driver_str == "cuda") {
    config.backend = "cuda";
  } else if (driver_str == "hip") {
    config.backend = "hip";
  } else if (driver_str == "local-task" || driver_str == "local-sync") {
    config.backend = "llvm-cpu";
  } else {
    return Ort::Status(("IREE EP: Unknown driver: " + driver_str).c_str(),
                       ORT_INVALID_ARGUMENT)
        .release();
  }

  // Require target_arch for non-CPU devices.
  OrtHardwareDeviceType device_type =
      factory->ort_api.HardwareDevice_Type(&hardware_device);
  if (device_type != OrtHardwareDeviceType_CPU && config.target_arch.empty()) {
    return Ort::Status(
               "IREE EP: 'target_arch' option must be specified for non-CPU "
               "devices",
               ORT_INVALID_ARGUMENT)
        .release();
  }

  ORT_CXX_LOGF_NOEXCEPT(factory->logger_, ORT_LOGGING_LEVEL_INFO,
                        "IREE EP: Creating EP with driver='%s', "
                        "device_path='%s', target_arch='%s', opt_level='%s'",
                        driver_name, device_path, config.target_arch.c_str(),
                        config.opt_level.c_str());

  // Build device URI and create HAL device.
  std::string device_uri = driver_str + "://" + device_path;
  HalDevicePtr hal_device;
  iree_status_t status = iree_hal_create_device(
      iree_runtime_instance_driver_registry(factory->IreeInstance()),
      iree_make_string_view(device_uri.data(), device_uri.size()),
      iree_allocator_system(), hal_device.ForOutput());
  if (!iree_status_is_ok(status)) {
    return IreeStatusToOrtStatus(status);
  }

  // Create the EP instance with the device.
  auto ep_instance = std::make_unique<IreeEp>(
      *factory, factory->ep_name_, config, *logger, std::move(hal_device));

  *ep = ep_instance.release();
  return nullptr;
}

/*static*/
void ORT_API_CALL IreeEpFactory::ReleaseEpImpl(OrtEpFactory* /*this_ptr*/,
                                               OrtEp* ep) noexcept {
  delete static_cast<IreeEp*>(ep);
}

/*static*/
OrtStatus* ORT_API_CALL IreeEpFactory::CreateAllocatorImpl(
    OrtEpFactory* /*this_ptr*/, const OrtMemoryInfo* /*memory_info*/,
    const OrtKeyValuePairs* /*allocator_options*/,
    OrtAllocator** allocator) noexcept {
  // No custom allocator yet
  *allocator = nullptr;
  return nullptr;
}

/*static*/
void ORT_API_CALL IreeEpFactory::ReleaseAllocatorImpl(
    OrtEpFactory* /*this_ptr*/, OrtAllocator* /*allocator*/) noexcept {
  // Nothing to do
}

/*static*/
OrtStatus* ORT_API_CALL IreeEpFactory::CreateDataTransferImpl(
    OrtEpFactory* /*this_ptr*/, OrtDataTransferImpl** data_transfer) noexcept {
  // No custom data transfer yet
  *data_transfer = nullptr;
  return nullptr;
}

/*static*/
bool ORT_API_CALL
IreeEpFactory::IsStreamAwareImpl(const OrtEpFactory* /*this_ptr*/) noexcept {
  // Not stream aware yet (synchronous execution only)
  return false;
}

/*static*/
OrtStatus* ORT_API_CALL IreeEpFactory::CreateSyncStreamForDeviceImpl(
    OrtEpFactory* /*this_ptr*/, const OrtMemoryDevice* /*memory_device*/,
    const OrtKeyValuePairs* /*stream_options*/,
    OrtSyncStreamImpl** stream) noexcept {
  // Not stream aware
  *stream = nullptr;
  return nullptr;
}

}  // namespace iree_onnx_ep
