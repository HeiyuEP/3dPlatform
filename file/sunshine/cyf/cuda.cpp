/**
 * @file src/platform/linux/cuda.cpp
 * @brief Definitions for CUDA encoding.
 */
#include <bitset>
#include <fcntl.h>
#include <filesystem>
#include <thread>

#include <NvFBC.h>
#include <ffnvcodec/dynlink_loader.h>
#include <opencv2/opencv.hpp>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/imgutils.h>
}

#include "cuda.h"
#include "graphics.h"
#include "src/logging.h"
#include "src/utility.h"
#include "src/video.h"
#include "wayland.h"

#include <iostream>
#include <cstdlib>
#include <string>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <chrono>
#include <atomic>
#include "GenerateDepthImg.h"
#include "securec.h"
#include "string.h"

#define SUNSHINE_STRINGVIEW_HELPER(x) x##sv
#define SUNSHINE_STRINGVIEW(x) SUNSHINE_STRINGVIEW_HELPER(x)
#define SUNSHINE_HARDWARD_ENCODE
#define LATENCY_TEST false

#define CU_CHECK(x, y) \
  if (check((x), SUNSHINE_STRINGVIEW(y ": "))) return -1

#define CU_CHECK_IGNORE(x, y) \
  check((x), SUNSHINE_STRINGVIEW(y ": "))

namespace fs = std::filesystem;

#ifdef SUNSHINE_HARDWARD_ENCODE
  extern int DIBRFlagTest(uint8_t* colorData, uint8_t* depth_data, uint8_t** resultImg, bool outputLeft, uint8_t flag);
  extern int MAX_CAPTURE_FRAME_COUNT;
  static int depth_to_dibr_img_count = 0;
  int src_rgb_image_width = 1920;
  int src_rgb_image_height = 1080;
  int src_rgba_image_channel = 4;
#endif

using namespace std::literals;
namespace cuda {
  constexpr auto cudaDevAttrMaxThreadsPerBlock = (CUdevice_attribute) 1;
  constexpr auto cudaDevAttrMaxThreadsPerMultiProcessor = (CUdevice_attribute) 39;

  void
  pass_error(const std::string_view &sv, const char *name, const char *description) {
    BOOST_LOG(error) << sv << name << ':' << description;
  }

  void
  cff(CudaFunctions *cf) {
    cuda_free_functions(&cf);
  }

  using cdf_t = util::safe_ptr<CudaFunctions, cff>;

  static cdf_t cdf;

  inline static int
  check(CUresult result, const std::string_view &sv) {
    if (result != CUDA_SUCCESS) {
      const char *name;
      const char *description;

      cdf->cuGetErrorName(result, &name);
      cdf->cuGetErrorString(result, &description);

      BOOST_LOG(error) << sv << name << ':' << description;
      return -1;
    }

    return 0;
  }

  void
  freeStream(CUstream stream) {
    CU_CHECK_IGNORE(cdf->cuStreamDestroy(stream), "Couldn't destroy cuda stream");
  }

  void
  unregisterResource(CUgraphicsResource resource) {
    CU_CHECK_IGNORE(cdf->cuGraphicsUnregisterResource(resource), "Couldn't unregister resource");
  }

  using registered_resource_t = util::safe_ptr<CUgraphicsResource_st, unregisterResource>;

  class img_t: public platf::img_t {
  public:
    tex_t tex;
  };

#ifdef SUNSHINE_HARDWARD_ENCODE
  struct img_queue_data {
    img_t* imgT;
    uint8_t* data;
    std::chrono::time_point<std::chrono::_V2::system_clock, std::chrono::duration<long int, std::ratio<1, 1000000000> > > timestamp;
  };

  std::queue<img_queue_data> src_images;
  std::queue<img_queue_data> depth_images;
  std::queue<img_queue_data> src_rgb_images;
  std::mutex mtx_src_images, mtx_depth_images;
  std::condition_variable cv_src_images, cv_depth_images;
  std::atomic<bool> running(true); // 线程运行标志
#endif

  int
  init() {
    auto status = cuda_load_functions(&cdf, nullptr);
    if (status) {
      BOOST_LOG(error) << "Couldn't load cuda: "sv << status;

      return -1;
    }

    CU_CHECK(cdf->cuInit(0), "Couldn't initialize cuda");

    return 0;
  }

  class cuda_t: public platf::avcodec_encode_device_t {
  public:
    int
    init(int in_width, int in_height) {
      if (!cdf) {
        BOOST_LOG(warning) << "cuda not initialized"sv;
        return -1;
      }

      data = (void *) 0x1;

      width = in_width;
      height = in_height;

      return 0;
    }

    int
    set_frame(AVFrame *frame, AVBufferRef *hw_frames_ctx) override {
      this->hwframe.reset(frame);
      this->frame = frame;

      auto hwframe_ctx = (AVHWFramesContext *) hw_frames_ctx->data;
      if (hwframe_ctx->sw_format != AV_PIX_FMT_NV12) {
        BOOST_LOG(error) << "cuda::cuda_t doesn't support any format other than AV_PIX_FMT_NV12"sv;
        return -1;
      }

      if (!frame->buf[0]) {
        if (av_hwframe_get_buffer(hw_frames_ctx, frame, 0)) {
          BOOST_LOG(error) << "Couldn't get hwframe for NVENC"sv;
          return -1;
        }
      }

      auto cuda_ctx = (AVCUDADeviceContext *) hwframe_ctx->device_ctx->hwctx;

      stream = make_stream();
      if (!stream) {
        return -1;
      }

      cuda_ctx->stream = stream.get();

      auto sws_opt = sws_t::make(width, height, frame->width, frame->height, width * 4);
      if (!sws_opt) {
        return -1;
      }

      sws = std::move(*sws_opt);

      linear_interpolation = width != frame->width || height != frame->height;

      return 0;
    }

    void
    apply_colorspace() override {
      sws.apply_colorspace(colorspace);

      auto tex = tex_t::make(height, width * 4);
      if (!tex) {
        return;
      }

      // The default green color is ugly.
      // Update the background color
      platf::img_t img;
      img.width = width;
      img.height = height;
      img.pixel_pitch = 4;
      img.row_pitch = img.width * img.pixel_pitch;

      std::vector<std::uint8_t> image_data;
      image_data.resize(img.row_pitch * img.height);

      img.data = image_data.data();

      if (sws.load_ram(img, tex->array)) {
        return;
      }

      sws.convert(frame->data[0], frame->data[1], frame->linesize[0], frame->linesize[1], tex->texture.linear, stream.get(), { frame->width, frame->height, 0, 0 });
    }

    cudaTextureObject_t
    tex_obj(const tex_t &tex) const {
      return linear_interpolation ? tex.texture.linear : tex.texture.point;
    }

    stream_t stream;
    frame_t hwframe;

    int width, height;

    // When height and width don't change, it's not necessary to use linear interpolation
    bool linear_interpolation;

    sws_t sws;
  };

  class cuda_ram_t: public cuda_t {
  public:
    int
    convert(platf::img_t &img) override {
      return sws.load_ram(img, tex.array) || sws.convert(frame->data[0], frame->data[1], frame->linesize[0], frame->linesize[1], tex_obj(tex), stream.get());
    }

    int
    set_frame(AVFrame *frame, AVBufferRef *hw_frames_ctx) {
      if (cuda_t::set_frame(frame, hw_frames_ctx)) {
        return -1;
      }

      auto tex_opt = tex_t::make(height, width * 4);
      if (!tex_opt) {
        return -1;
      }

      tex = std::move(*tex_opt);

      return 0;
    }

    tex_t tex;
  };

  class cuda_vram_t: public cuda_t {
  public:
    int
    convert(platf::img_t &img) override {
      return sws.convert(frame->data[0], frame->data[1], frame->linesize[0], frame->linesize[1], tex_obj(((img_t *) &img)->tex), stream.get());
    }
  };

  /**
   * @brief Opens the DRM device associated with the CUDA device index.
   * @param index CUDA device index to open.
   * @return File descriptor or -1 on failure.
   */
  file_t
  open_drm_fd_for_cuda_device(int index) {
    CUdevice device;
    CU_CHECK(cdf->cuDeviceGet(&device, index), "Couldn't get CUDA device");

    // There's no way to directly go from CUDA to a DRM device, so we'll
    // use sysfs to look up the DRM device name from the PCI ID.
    std::array<char, 13> pci_bus_id;
    CU_CHECK(cdf->cuDeviceGetPCIBusId(pci_bus_id.data(), pci_bus_id.size(), device), "Couldn't get CUDA device PCI bus ID");
    BOOST_LOG(debug) << "Found CUDA device with PCI bus ID: "sv << pci_bus_id.data();

    // Linux uses lowercase hexadecimal while CUDA uses uppercase
    std::transform(pci_bus_id.begin(), pci_bus_id.end(), pci_bus_id.begin(),
      [](char c) { return std::tolower(c); });

    // Look for the name of the primary node in sysfs
    try {
      char sysfs_path[PATH_MAX];
      std::snprintf(sysfs_path, sizeof(sysfs_path), "/sys/bus/pci/devices/%s/drm", pci_bus_id.data());
      fs::path sysfs_dir { sysfs_path };
      for (auto &entry : fs::directory_iterator { sysfs_dir }) {
        auto file = entry.path().filename();
        auto filestring = file.generic_string();
        if (std::string_view { filestring }.substr(0, 4) != "card"sv) {
          continue;
        }

        BOOST_LOG(debug) << "Found DRM primary node: "sv << filestring;

        fs::path dri_path { "/dev/dri"sv };
        auto device_path = dri_path / file;
        return open(device_path.c_str(), O_RDWR);
      }
    }
    catch (const std::filesystem::filesystem_error &err) {
      BOOST_LOG(error) << "Failed to read sysfs: "sv << err.what();
    }

    BOOST_LOG(error) << "Unable to find DRM device with PCI bus ID: "sv << pci_bus_id.data();
    return -1;
  }

  class gl_cuda_vram_t: public platf::avcodec_encode_device_t {
  public:
    /**
     * @brief Initialize the GL->CUDA encoding device.
     * @param in_width Width of captured frames.
     * @param in_height Height of captured frames.
     * @param offset_x Offset of content in captured frame.
     * @param offset_y Offset of content in captured frame.
     * @return 0 on success or -1 on failure.
     */
    int
    init(int in_width, int in_height, int offset_x, int offset_y) {
      // This must be non-zero to tell the video core that it's a hardware encoding device.
      data = (void *) 0x1;

      // TODO: Support more than one CUDA device
      file = std::move(open_drm_fd_for_cuda_device(0));
      if (file.el < 0) {
        char string[1024];
        BOOST_LOG(error) << "Couldn't open DRM FD for CUDA device: "sv << strerror_r(errno, string, sizeof(string));
        return -1;
      }

      gbm.reset(gbm::create_device(file.el));
      if (!gbm) {
        BOOST_LOG(error) << "Couldn't create GBM device: ["sv << util::hex(eglGetError()).to_string_view() << ']';
        return -1;
      }

      display = egl::make_display(gbm.get());
      if (!display) {
        return -1;
      }

      auto ctx_opt = egl::make_ctx(display.get());
      if (!ctx_opt) {
        return -1;
      }

      ctx = std::move(*ctx_opt);

      width = in_width;
      height = in_height;

      sequence = 0;

      this->offset_x = offset_x;
      this->offset_y = offset_y;

      return 0;
    }

    /**
     * @brief Initialize color conversion into target CUDA frame.
     * @param frame Destination CUDA frame to write into.
     * @param hw_frames_ctx_buf FFmpeg hardware frame context.
     * @return 0 on success or -1 on failure.
     */
    int
    set_frame(AVFrame *frame, AVBufferRef *hw_frames_ctx_buf) override {
      this->hwframe.reset(frame);
      this->frame = frame;

      if (!frame->buf[0]) {
        if (av_hwframe_get_buffer(hw_frames_ctx_buf, frame, 0)) {
          BOOST_LOG(error) << "Couldn't get hwframe for VAAPI"sv;
          return -1;
        }
      }

      auto hw_frames_ctx = (AVHWFramesContext *) hw_frames_ctx_buf->data;
      sw_format = hw_frames_ctx->sw_format;

      auto nv12_opt = egl::create_target(frame->width, frame->height, sw_format);
      if (!nv12_opt) {
        return -1;
      }

      auto sws_opt = egl::sws_t::make(width, height, frame->width, frame->height, sw_format);
      if (!sws_opt) {
        return -1;
      }

      this->sws = std::move(*sws_opt);
      this->nv12 = std::move(*nv12_opt);

      auto cuda_ctx = (AVCUDADeviceContext *) hw_frames_ctx->device_ctx->hwctx;

      stream = make_stream();
      if (!stream) {
        return -1;
      }

      cuda_ctx->stream = stream.get();

      CU_CHECK(cdf->cuGraphicsGLRegisterImage(&y_res, nv12->tex[0], GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY),
        "Couldn't register Y plane texture");
      CU_CHECK(cdf->cuGraphicsGLRegisterImage(&uv_res, nv12->tex[1], GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY),
        "Couldn't register UV plane texture");

      return 0;
    }

    /**
     * @brief Convert the captured image into the target CUDA frame.
     * @param img Captured screen image.
     * @return 0 on success or -1 on failure.
     */
    int
    convert(platf::img_t &img) override {
      auto &descriptor = (egl::img_descriptor_t &) img;

      if (descriptor.sequence == 0) {
        // For dummy images, use a blank RGB texture instead of importing a DMA-BUF
        rgb = egl::create_blank(img);
      }
      else if (descriptor.sequence > sequence) {
        sequence = descriptor.sequence;

        rgb = egl::rgb_t {};

        auto rgb_opt = egl::import_source(display.get(), descriptor.sd);

        if (!rgb_opt) {
          return -1;
        }

        rgb = std::move(*rgb_opt);
      }

      // Perform the color conversion and scaling in GL
      sws.load_vram(descriptor, offset_x, offset_y, rgb->tex[0]);
      sws.convert(nv12->buf);

      auto fmt_desc = av_pix_fmt_desc_get(sw_format);

      // Map the GL textures to read for CUDA
      CUgraphicsResource resources[2] = { y_res.get(), uv_res.get() };
      CU_CHECK(cdf->cuGraphicsMapResources(2, resources, stream.get()), "Couldn't map GL textures in CUDA");

      // Copy from the GL textures to the target CUDA frame
      for (int i = 0; i < 2; i++) {
        CUDA_MEMCPY2D cpy = {};
        cpy.srcMemoryType = CU_MEMORYTYPE_ARRAY;
        CU_CHECK(cdf->cuGraphicsSubResourceGetMappedArray(&cpy.srcArray, resources[i], 0, 0), "Couldn't get mapped plane array");

        cpy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
        cpy.dstDevice = (CUdeviceptr) frame->data[i];
        cpy.dstPitch = frame->linesize[i];
        cpy.WidthInBytes = (frame->width * fmt_desc->comp[i].step) >> (i ? fmt_desc->log2_chroma_w : 0);
        cpy.Height = frame->height >> (i ? fmt_desc->log2_chroma_h : 0);

        CU_CHECK_IGNORE(cdf->cuMemcpy2DAsync(&cpy, stream.get()), "Couldn't copy texture to CUDA frame");
      }

      // Unmap the textures to allow modification from GL again
      CU_CHECK(cdf->cuGraphicsUnmapResources(2, resources, stream.get()), "Couldn't unmap GL textures from CUDA");
      return 0;
    }

    /**
     * @brief Configures shader parameters for the specified colorspace.
     */
    void
    apply_colorspace() override {
      sws.apply_colorspace(colorspace);
    }

    file_t file;
    gbm::gbm_t gbm;
    egl::display_t display;
    egl::ctx_t ctx;

    // This must be destroyed before display_t
    stream_t stream;
    frame_t hwframe;

    egl::sws_t sws;
    egl::nv12_t nv12;
    AVPixelFormat sw_format;

    int width, height;

    std::uint64_t sequence;
    egl::rgb_t rgb;

    registered_resource_t y_res;
    registered_resource_t uv_res;

    int offset_x, offset_y;
  };

  std::unique_ptr<platf::avcodec_encode_device_t>
  make_avcodec_encode_device(int width, int height, bool vram) {
    if (init()) {
      return nullptr;
    }

    std::unique_ptr<cuda_t> cuda;

    if (vram) {
      cuda = std::make_unique<cuda_vram_t>();
    }
    else {
      cuda = std::make_unique<cuda_ram_t>();
    }

    if (cuda->init(width, height)) {
      return nullptr;
    }

    return cuda;
  }

  /**
   * @brief Create a GL->CUDA encoding device for consuming captured dmabufs.
   * @param width Width of captured frames.
   * @param height Height of captured frames.
   * @param offset_x Offset of content in captured frame.
   * @param offset_y Offset of content in captured frame.
   * @return FFmpeg encoding device context.
   */
  std::unique_ptr<platf::avcodec_encode_device_t>
  make_avcodec_gl_encode_device(int width, int height, int offset_x, int offset_y) {
    if (init()) {
      return nullptr;
    }

    auto cuda = std::make_unique<gl_cuda_vram_t>();

    if (cuda->init(width, height, offset_x, offset_y)) {
      return nullptr;
    }

    return cuda;
  }

  namespace nvfbc {
    static PNVFBCCREATEINSTANCE createInstance {};
    static NVFBC_API_FUNCTION_LIST func { NVFBC_VERSION };

    static constexpr inline NVFBC_BOOL
    nv_bool(bool b) {
      return b ? NVFBC_TRUE : NVFBC_FALSE;
    }

    static void *handle { nullptr };
    int
    init() {
      static bool funcs_loaded = false;

      if (funcs_loaded) return 0;

      if (!handle) {
        handle = dyn::handle({ "libnvidia-fbc.so.1", "libnvidia-fbc.so" });
        if (!handle) {
          return -1;
        }
      }

      std::vector<std::tuple<dyn::apiproc *, const char *>> funcs {
        { (dyn::apiproc *) &createInstance, "NvFBCCreateInstance" },
      };

      if (dyn::load(handle, funcs)) {
        dlclose(handle);
        handle = nullptr;

        return -1;
      }

      auto status = cuda::nvfbc::createInstance(&cuda::nvfbc::func);
      if (status) {
        BOOST_LOG(error) << "Unable to create NvFBC instance"sv;

        dlclose(handle);
        handle = nullptr;
        return -1;
      }

      funcs_loaded = true;
      return 0;
    }

    class ctx_t {
    public:
      ctx_t(NVFBC_SESSION_HANDLE handle) {
        NVFBC_BIND_CONTEXT_PARAMS params { NVFBC_BIND_CONTEXT_PARAMS_VER };

        if (func.nvFBCBindContext(handle, &params)) {
          BOOST_LOG(error) << "Couldn't bind NvFBC context to current thread: " << func.nvFBCGetLastErrorStr(handle);
        }

        this->handle = handle;
      }

      ~ctx_t() {
        NVFBC_RELEASE_CONTEXT_PARAMS params { NVFBC_RELEASE_CONTEXT_PARAMS_VER };
        if (func.nvFBCReleaseContext(handle, &params)) {
          BOOST_LOG(error) << "Couldn't release NvFBC context from current thread: " << func.nvFBCGetLastErrorStr(handle);
        }
      }

      NVFBC_SESSION_HANDLE handle;
    };

    class handle_t {
      enum flag_e {
        SESSION_HANDLE,
        SESSION_CAPTURE,
        MAX_FLAGS,
      };

    public:
      handle_t() = default;
      handle_t(handle_t &&other):
          handle_flags { other.handle_flags }, handle { other.handle } {
        other.handle_flags.reset();
      }

      handle_t &
      operator=(handle_t &&other) {
        std::swap(handle_flags, other.handle_flags);
        std::swap(handle, other.handle);

        return *this;
      }

      static std::optional<handle_t>
      make() {
        NVFBC_CREATE_HANDLE_PARAMS params { NVFBC_CREATE_HANDLE_PARAMS_VER };

        // Set privateData to allow NvFBC on consumer NVIDIA GPUs.
        // Based on https://github.com/keylase/nvidia-patch/blob/3193b4b1cea91527bf09ea9b8db5aade6a3f3c0a/win/nvfbcwrp/nvfbcwrp_main.cpp#L23-L25 .
        const unsigned int MAGIC_PRIVATE_DATA[4] = { 0xAEF57AC5, 0x401D1A39, 0x1B856BBE, 0x9ED0CEBA };
        params.privateData = MAGIC_PRIVATE_DATA;
        params.privateDataSize = sizeof(MAGIC_PRIVATE_DATA);

        handle_t handle;
        auto status = func.nvFBCCreateHandle(&handle.handle, &params);
        if (status) {
          BOOST_LOG(error) << "Failed to create session: "sv << handle.last_error();

          return std::nullopt;
        }

        handle.handle_flags[SESSION_HANDLE] = true;

        return handle;
      }

      const char *
      last_error() {
        return func.nvFBCGetLastErrorStr(handle);
      }

      std::optional<NVFBC_GET_STATUS_PARAMS>
      status() {
        NVFBC_GET_STATUS_PARAMS params { NVFBC_GET_STATUS_PARAMS_VER };

        auto status = func.nvFBCGetStatus(handle, &params);
        if (status) {
          BOOST_LOG(error) << "Failed to get NvFBC status: "sv << last_error();

          return std::nullopt;
        }

        return params;
      }

      int
      capture(NVFBC_CREATE_CAPTURE_SESSION_PARAMS &capture_params) {
        if (func.nvFBCCreateCaptureSession(handle, &capture_params)) {
          BOOST_LOG(error) << "Failed to start capture session: "sv << last_error();
          return -1;
        }

        handle_flags[SESSION_CAPTURE] = true;

        NVFBC_TOCUDA_SETUP_PARAMS setup_params {
          NVFBC_TOCUDA_SETUP_PARAMS_VER,
          NVFBC_BUFFER_FORMAT_BGRA,
        };

        if (func.nvFBCToCudaSetUp(handle, &setup_params)) {
          BOOST_LOG(error) << "Failed to setup cuda interop with nvFBC: "sv << last_error();
          return -1;
        }
        return 0;
      }

      int
      stop() {
        if (!handle_flags[SESSION_CAPTURE]) {
          return 0;
        }

        NVFBC_DESTROY_CAPTURE_SESSION_PARAMS params { NVFBC_DESTROY_CAPTURE_SESSION_PARAMS_VER };

        if (func.nvFBCDestroyCaptureSession(handle, &params)) {
          BOOST_LOG(error) << "Couldn't destroy capture session: "sv << last_error();

          return -1;
        }

        handle_flags[SESSION_CAPTURE] = false;

        return 0;
      }

      int
      reset() {
        if (!handle_flags[SESSION_HANDLE]) {
          return 0;
        }

        stop();

        NVFBC_DESTROY_HANDLE_PARAMS params { NVFBC_DESTROY_HANDLE_PARAMS_VER };

        ctx_t ctx { handle };
        if (func.nvFBCDestroyHandle(handle, &params)) {
          BOOST_LOG(error) << "Couldn't destroy session handle: "sv << func.nvFBCGetLastErrorStr(handle);
        }

        handle_flags[SESSION_HANDLE] = false;

        return 0;
      }

      ~handle_t() {
        reset();
      }

      std::bitset<MAX_FLAGS> handle_flags;

      NVFBC_SESSION_HANDLE handle;
    };

    class display_t: public platf::display_t {
    public:
      int
      init(const std::string_view &display_name, const ::video::config_t &config) {
        auto handle = handle_t::make();
        if (!handle) {
          return -1;
        }

        ctx_t ctx { handle->handle };

        auto status_params = handle->status();
        if (!status_params) {
          return -1;
        }

        int streamedMonitor = -1;
        if (!display_name.empty()) {
          if (status_params->bXRandRAvailable) {
            auto monitor_nr = util::from_view(display_name);

            if (monitor_nr < 0 || monitor_nr >= status_params->dwOutputNum) {
              BOOST_LOG(warning) << "Can't stream monitor ["sv << monitor_nr << "], it needs to be between [0] and ["sv << status_params->dwOutputNum - 1 << "], defaulting to virtual desktop"sv;
            }
            else {
              streamedMonitor = monitor_nr;
            }
          }
          else {
            BOOST_LOG(warning) << "XrandR not available, streaming entire virtual desktop"sv;
          }
        }

        delay = std::chrono::nanoseconds { 1s } / config.framerate;

        capture_params = NVFBC_CREATE_CAPTURE_SESSION_PARAMS { NVFBC_CREATE_CAPTURE_SESSION_PARAMS_VER };

        capture_params.eCaptureType = NVFBC_CAPTURE_SHARED_CUDA;
        capture_params.bDisableAutoModesetRecovery = nv_bool(true);

        capture_params.dwSamplingRateMs = 1000 /* ms */ / config.framerate;

        if (streamedMonitor != -1) {
          auto &output = status_params->outputs[streamedMonitor];

          width = output.trackedBox.w;
          height = output.trackedBox.h;
          offset_x = output.trackedBox.x;
          offset_y = output.trackedBox.y;

          capture_params.eTrackingType = NVFBC_TRACKING_OUTPUT;
          capture_params.dwOutputId = output.dwId;
        }
        else {
          capture_params.eTrackingType = NVFBC_TRACKING_SCREEN;

          width = status_params->screenSize.w;
          height = status_params->screenSize.h;
        }

        env_width = status_params->screenSize.w;
        env_height = status_params->screenSize.h;

        this->handle = std::move(*handle);
        return 0;
      }

      void clear_queue(std::queue<img_queue_data> &queue) {
        while (!queue.empty()) {
          queue.pop();
        }
      }

      platf::capture_e
      capture(const push_captured_image_cb_t &push_captured_image_cb, const pull_free_image_cb_t &pull_free_image_cb, bool *cursor) override {
        auto next_frame = std::chrono::steady_clock::now();

        {
          // We must create at least one texture on this thread before calling NvFBCToCudaSetUp()
          // Otherwise it fails with "Unable to register an OpenGL buffer to a CUDA resource (result: 201)" message
          std::shared_ptr<platf::img_t> img_dummy;
          pull_free_image_cb(img_dummy);
        }

        // Force display_t::capture to initialize handle_t::capture
        cursor_visible = !*cursor;

        ctx_t ctx { handle.handle };
        auto fg = util::fail_guard([&]() {
          handle.reset();
        });

        sleep_overshoot_logger.reset();

#ifdef SUNSHINE_HARDWARD_ENCODE
        BOOST_LOG(info) << "Starting threads.";
        std::thread get_depth_thread(get_depth_img);
        get_depth_thread.detach();

        std::thread dibr_thread(get_3D_img);
        dibr_thread.detach();
#endif

        while (true) {
          BOOST_LOG(info) << "pre-snapshot.";
          auto now = std::chrono::steady_clock::now();
          if (next_frame > now) {
            std::this_thread::sleep_for(next_frame - now);
            sleep_overshoot_logger.first_point(next_frame);
            sleep_overshoot_logger.second_point_now_and_log();
          }

          next_frame += delay;
          if (next_frame < now) {  // some major slowdown happened; we couldn't keep up
            next_frame = now + delay;
          }
          BOOST_LOG(info) << "ready to snap shot..";

          std::shared_ptr<platf::img_t> img_out;
          auto status = snapshot(pull_free_image_cb, img_out, 150ms, *cursor);

          switch (status) {
            case platf::capture_e::reinit:
            case platf::capture_e::error:
            case platf::capture_e::interrupted:
              return status;
            case platf::capture_e::timeout:
              if (!push_captured_image_cb(std::move(img_out), false)) {
                return platf::capture_e::ok;
              }
              break;
            case platf::capture_e::ok:
              if (!push_captured_image_cb(std::move(img_out), true)) {
                return platf::capture_e::ok;
              }
              break;
            default:
              BOOST_LOG(error) << "Unrecognized capture status ["sv << (int) status << ']';
              return status;
          }
        }
#ifdef SUNSHINE_HARDWARD_ENCODE
        clear_queue(src_images);
        clear_queue(depth_images);
        clear_queue(src_rgb_images); 
        BOOST_LOG(info) << "clear_queue.";
        
        running =false;
        cv_src_images.notify_all();
#endif
        return platf::capture_e::ok;
      }

      // Reinitialize the capture session.
      platf::capture_e
      reinit(bool cursor) {
        if (handle.stop()) {
          return platf::capture_e::error;
        }

        cursor_visible = cursor;
        if (cursor) {
          capture_params.bPushModel = nv_bool(false);
          capture_params.bWithCursor = nv_bool(true);
          capture_params.bAllowDirectCapture = nv_bool(false);
        }
        else {
          capture_params.bPushModel = nv_bool(true);
          capture_params.bWithCursor = nv_bool(false);
          capture_params.bAllowDirectCapture = nv_bool(true);
        }

        if (handle.capture(capture_params)) {
          return platf::capture_e::error;
        }

        // If trying to capture directly, test if it actually does.
        if (capture_params.bAllowDirectCapture) {
          CUdeviceptr device_ptr;
          NVFBC_FRAME_GRAB_INFO info;

          NVFBC_TOCUDA_GRAB_FRAME_PARAMS grab {
            NVFBC_TOCUDA_GRAB_FRAME_PARAMS_VER,
            NVFBC_TOCUDA_GRAB_FLAGS_NOWAIT,
            &device_ptr,
            &info,
            0,
          };

          // Direct Capture may fail the first few times, even if it's possible
          for (int x = 0; x < 3; ++x) {
            if (auto status = func.nvFBCToCudaGrabFrame(handle.handle, &grab)) {
              if (status == NVFBC_ERR_MUST_RECREATE) {
                return platf::capture_e::reinit;
              }

              BOOST_LOG(error) << "Couldn't capture nvFramebuffer: "sv << handle.last_error();

              return platf::capture_e::error;
            }

            if (info.bDirectCapture) {
              break;
            }

            BOOST_LOG(debug) << "Direct capture failed attempt ["sv << x << ']';
          }

          if (!info.bDirectCapture) {
            BOOST_LOG(debug) << "Direct capture failed, trying the extra copy method"sv;
            // Direct capture failed
            capture_params.bPushModel = nv_bool(false);
            capture_params.bWithCursor = nv_bool(false);
            capture_params.bAllowDirectCapture = nv_bool(false);

            if (handle.stop() || handle.capture(capture_params)) {
              return platf::capture_e::error;
            }
          }
        }

        return platf::capture_e::ok;
      }

      platf::capture_e
      snapshot(
        const pull_free_image_cb_t &pull_free_image_cb,
        std::shared_ptr<platf::img_t> &img_out, 
        std::chrono::milliseconds timeout, 
        bool cursor) {
        BOOST_LOG(info) << "snapshot";
        if (cursor != cursor_visible) {
          auto status = reinit(cursor);
          if (status != platf::capture_e::ok) {
            return status;
          }
        }

        CUdeviceptr device_ptr;
        NVFBC_FRAME_GRAB_INFO info;

        NVFBC_TOCUDA_GRAB_FRAME_PARAMS grab {
          NVFBC_TOCUDA_GRAB_FRAME_PARAMS_VER,
          NVFBC_TOCUDA_GRAB_FLAGS_NOWAIT,
          &device_ptr,
          &info,
          (std::uint32_t) timeout.count(),
        };

        if (auto status = func.nvFBCToCudaGrabFrame(handle.handle, &grab)) {
          if (status == NVFBC_ERR_MUST_RECREATE) {
            return platf::capture_e::reinit;
          }

          BOOST_LOG(error) << "Couldn't capture nvFramebuffer: "sv << handle.last_error();
          return platf::capture_e::error;
        }

        if (!pull_free_image_cb(img_out)) {
          return platf::capture_e::interrupted;
        }
        auto img = (img_t *) img_out.get();

#ifdef SUNSHINE_HARDWARD_ENCODE
        uint32_t frame_size = img->height * img->width * img->pixel_pitch;
        uint8_t* src_image = new uint8_t[frame_size];
        if (src_image == nullptr) {
          BOOST_LOG(error) << "src_image alloc failed!";
          return platf::capture_e::error;
        }

        auto time_a = std::chrono::high_resolution_clock::now();
        img->tex.copyToHost((std::uint8_t *) device_ptr, src_image, frame_size);
        
        img_queue_data src_image_qd;
        src_image_qd.imgT = img;
        src_image_qd.data = src_image;
        src_image_qd.timestamp = time_a;

        {
          std::lock_guard<std::mutex> lock(mtx_src_images);
          if (src_images.size() >= MAX_CAPTURE_FRAME_COUNT) {
            delete src_images.front().data;
            src_images.pop();
          }
          src_images.push(src_image_qd);
        }
        cv_src_images.notify_one();
#else
        if (img->tex.copy((std::uint8_t *) device_ptr, img->height, img->row_pitch)) {
          return platf::capture_e::error;
        }
#endif
        return platf::capture_e::ok;
      }

      std::unique_ptr<platf::avcodec_encode_device_t>
      make_avcodec_encode_device(platf::pix_fmt_e pix_fmt) {
        return ::cuda::make_avcodec_encode_device(width, height, true);
      }

      std::shared_ptr<platf::img_t>
      alloc_img() override {
        auto img = std::make_shared<cuda::img_t>();

        img->data = nullptr;
        img->width = width;
        img->height = height;
        img->pixel_pitch = 4;
        img->row_pitch = img->width * img->pixel_pitch;

        auto tex_opt = tex_t::make(height, width * img->pixel_pitch);
        if (!tex_opt) {
          return nullptr;
        }

        img->tex = std::move(*tex_opt);

        return img;
      };

      int
      dummy_img(platf::img_t *) override {
        return 0;
      }

#ifdef SUNSHINE_HARDWARD_ENCODE
      static void get_depth_img() {
        BOOST_LOG(info) << "Starting get_depth_img thread.";
        int channel = 4;
        GenerateDepthImg* depth = new GenerateDepthImg(src_rgb_image_width,src_rgb_image_height,channel);
        if (depth == nullptr) {
          BOOST_LOG(error) << "GenerateDepthImg alloc failed!";
          return;
        }
        size_t depth_size = src_rgb_image_width*src_rgb_image_height;
        std::uint8_t *depth_data = new uint8_t[depth_size];
        if (depth_data == nullptr) {
          BOOST_LOG(error) << "depth_data alloc failed!";
          return;
        }
        size_t src_rgb_size = src_rgb_image_width*src_rgb_image_height*3;
        std::uint8_t *src_rgb_data = new uint8_t[src_rgb_size];
        if (src_rgb_data == nullptr) {
          BOOST_LOG(error) << "src_rgb_data alloc failed!";
          return;
        }

        std::chrono::duration<double, std::milli> time_cost_get_depth_img;        
        while (running) {
          auto time_a = std::chrono::high_resolution_clock::now();
          img_queue_data src_image_qd;        
          {
            std::unique_lock<std::mutex> lock(mtx_src_images);
            cv_src_images.wait(lock, [] { return !src_images.empty() || !running; });
            if (!src_images.empty()) {
              src_image_qd = src_images.front();
              src_images.pop();
            } else {
              continue;
            }
          }
          execute_depthV2_model(depth, src_image_qd, src_rgb_data, depth_data, depth_size, src_rgb_size);
          auto time_b = std::chrono::high_resolution_clock::now();
          time_cost_get_depth_img = time_b - time_a;
          BOOST_LOG(info) << "time_cost_get_depth_img: " << time_cost_get_depth_img.count() << "ms";
        }
        if (depth_data != nullptr) {
          delete depth_data;
          depth_data = nullptr;
        }
        if (src_rgb_data != nullptr) {
          delete src_rgb_data;
          src_rgb_data = nullptr;
        }
        delete depth;
      }

      static void
      execute_depthV2_model(GenerateDepthImg* depth, img_queue_data src_image_qd, std::uint8_t *src_rgb_data, std::uint8_t *depth_data, size_t depth_size, size_t src_rgb_size) {
        uint8_t* depth_image = new uint8_t[depth_size];
        if (depth_image == nullptr) {
          BOOST_LOG(error) << "depth_image alloc failed!";
          return;
        }
        uint8_t* src_rgb_image = new uint8_t[src_rgb_size];
        if (src_rgb_image == nullptr) {
          BOOST_LOG(error) << "src_rgb_image alloc failed!";
          return;
        }

#if LATENCY_TEST
        depth->LatencyTest(src_image_qd.data, src_rgb_data, depth_data);
#else
        depth->Execute(src_image_qd.data, src_rgb_data, depth_data);
#endif
        delete src_image_qd.data;

        memcpy_s(depth_image, depth_size, depth_data, depth_size);
        memcpy_s(src_rgb_image, src_rgb_size, src_rgb_data, src_rgb_size);
        img_queue_data depth_image_qd;
        depth_image_qd.imgT = src_image_qd.imgT;
        depth_image_qd.data = depth_image;
        depth_image_qd.timestamp = src_image_qd.timestamp;

        img_queue_data src_rgb_image_qd;
        src_rgb_image_qd.imgT = src_image_qd.imgT;
        src_rgb_image_qd.data = src_rgb_image;
        src_rgb_image_qd.timestamp = src_image_qd.timestamp;

        BOOST_LOG(info) << "get_depth_img: Ready for Raise.";

        {
          std::lock_guard<std::mutex> lock(mtx_depth_images);
          if (depth_images.size() >= MAX_CAPTURE_FRAME_COUNT) {
            delete depth_images.front().data;
            depth_images.pop();
          }
          if (src_rgb_images.size() >= MAX_CAPTURE_FRAME_COUNT) {
            delete src_rgb_images.front().data;
            src_rgb_images.pop();
          }
          depth_images.push(depth_image_qd);
          src_rgb_images.push(src_rgb_image_qd);
        }
        cv_depth_images.notify_one();
        
        depth_to_dibr_img_count++;
        BOOST_LOG(info) << "get_depth_img: Raise completed.";
      }

      static void get_3D_img() {
        BOOST_LOG(info) << "get_3D_img thread.";
        int left_init_ret = DIBRFlagTest(nullptr, nullptr,nullptr,1,0);
        int right_init_ret = DIBRFlagTest(nullptr, nullptr,nullptr,0,0);
        if (left_init_ret || right_init_ret) {
          BOOST_LOG(error) << "DIBRFlagTest return fail.";
          return;
        }
        std::chrono::duration<double, std::milli> time_cost_get_3D_img;
        std::chrono::duration<double, std::milli> time_cost_whole_pipeline;
        while (running) {
          img_queue_data depth_image_qd;
          img_queue_data src_rgb_image_qd;
          auto time_a = std::chrono::high_resolution_clock::now();       
          {
            std::unique_lock<std::mutex> lock(mtx_depth_images);
            cv_depth_images.wait(lock, [] { return !depth_images.empty() || !running; });
            if (!depth_images.empty()) {
              time_a = std::chrono::high_resolution_clock::now();
              depth_image_qd = depth_images.front();
              depth_images.pop();
              src_rgb_image_qd = src_rgb_images.front();
              src_rgb_images.pop();
            } else {
              continue;
            }
          }
          depth_to_dibr_img_count--;
#if LATENCY_TEST
          get_3D_img_no_render(src_rgb_image_qd, depth_image_qd);
#else
          get_3D_img_do_render(src_rgb_image_qd, depth_image_qd);
#endif          
          delete src_rgb_image_qd.data;
          delete depth_image_qd.data;
          auto time_b = std::chrono::high_resolution_clock::now();
          time_cost_whole_pipeline = time_b - src_rgb_image_qd.timestamp;
          time_cost_get_3D_img = time_b - time_a;
          BOOST_LOG(info) << "time_cost_get_3D_img: " << time_cost_get_3D_img.count() << "ms";
          BOOST_LOG(info) << "time_cost_whole_pipeline: " << time_cost_whole_pipeline.count() << "ms";
        }
      }

      static void get_3D_img_no_render(img_queue_data src_rgb_image_qd, img_queue_data depth_image_qd) {
        cv::Mat imgRgb(src_rgb_image_height, src_rgb_image_width, CV_8UC3, src_rgb_image_qd.data);
        cv::Mat imgRgba;
        cv::cvtColor(imgRgb, imgRgba, cv::COLOR_BGR2BGRA);
        auto img = depth_image_qd.imgT;
        img->tex.copyToDevice(imgRgba.data, img->height, img->row_pitch);
      }

      static void get_3D_img_do_render(img_queue_data src_rgb_image_qd, img_queue_data depth_image_qd) {
        uint8_t* left_3d_result = nullptr;
        int retLeft = DIBRFlagTest(src_rgb_image_qd.data, depth_image_qd.data, &left_3d_result, 1, 1);
        uint8_t* right_3d_result = nullptr;
        int retRight = DIBRFlagTest(src_rgb_image_qd.data, depth_image_qd.data, &right_3d_result, 0, 1);
    
        if (retLeft == 0 && retRight == 0 && left_3d_result != nullptr && right_3d_result != nullptr) {
          get_result_img(left_3d_result, right_3d_result, depth_image_qd);
        }
      }

      static void get_result_img(uint8_t* left_3d_result, uint8_t* right_3d_result, img_queue_data depth_image_qd) {
        img_size single_width_img;
        single_width_img.width = src_rgb_image_width;
        single_width_img.height = src_rgb_image_height;
        single_width_img.channel = src_rgba_image_channel;

        uint8_t* concated_img_data = nullptr;
        concatenate_filp_images(left_3d_result, right_3d_result, single_width_img, &concated_img_data);
        uint8_t* resized_img_data = nullptr;
        half_width_image(concated_img_data, &resized_img_data, single_width_img);
        uint8_t* result_img = new uint8_t[src_rgb_image_width*src_rgb_image_height*src_rgba_image_channel];
        if (result_img == nullptr) {
          BOOST_LOG(error) << "result_img alloc failed!";
          return;
        }
        memcpy_s(result_img, src_rgb_image_width*src_rgb_image_height*src_rgba_image_channel, resized_img_data, src_rgb_image_width*src_rgb_image_height*src_rgba_image_channel);

        auto img = depth_image_qd.imgT;
        img->tex.copyToDevice(result_img, img->height, img->row_pitch);
        
        delete result_img;
        delete left_3d_result;
        delete right_3d_result;
        delete concated_img_data;
        delete resized_img_data;
        left_3d_result = nullptr;
        right_3d_result = nullptr;
        concated_img_data = nullptr;
        resized_img_data = nullptr;
      }

      static void concatenate_filp_images(uint8_t* img_data_left, uint8_t* img_data_right, img_size single_width_img, uint8_t** new_data) {
        int width = single_width_img.width;
        int height = single_width_img.height;
        int channels = single_width_img.channel;

        int new_width = width * 2; // 左右图拼接后的宽度*2
        *new_data = new uint8_t[new_width*height*channels];
        if (new_data == nullptr) {
          BOOST_LOG(error) << "concatenate new_data alloc failed!";
          return;
        }
        for (int y=0; y<height ; ++y) {
          int src_y = height - 1 - y;
          uint8_t* row_left = img_data_left + (src_y*width*channels);
          uint8_t* row_right = img_data_right + (src_y*width*channels);

          uint8_t* new_row = *new_data + (y*new_width*channels);
          memcpy_s(new_row, width*channels, row_left, width*channels);                   // 左半部分
          memcpy_s(new_row+(width*channels), width*channels, row_right, width*channels); // 右半部分
        }
      }

      static void half_width_image(uint8_t* input_data, uint8_t** output_data, img_size single_width_img) {
        int output_width = single_width_img.width;
        int input_width = output_width * 2; // 输入图片宽度是输出图片的2倍
        int input_height = single_width_img.height;
        int channels = single_width_img.channel;

        *output_data = new uint8_t[output_width*input_height*channels];
        if (output_data == nullptr) {
          BOOST_LOG(error) << "output_data alloc failed!";
          return;
        }

        float scale = static_cast<float>(input_width) / output_width;

        for (int y=0; y<input_height ; ++y) {
          for (int x=0;x<output_width;++x) {
            int src_x = static_cast<int>(x*scale);
            src_x = std::min(src_x, input_width-1);
            uint8_t* src_pixel = input_data + (y*input_width*channels)+(src_x*channels);
            uint8_t* dest_pixel = (*output_data) + (y*output_width*channels)+(x*channels);
            for (int c=0; c<channels ; ++c) {
              dest_pixel[c] = src_pixel[c];
            }
          }
        }
      }
#endif      

      std::chrono::nanoseconds delay;

      bool cursor_visible;
      handle_t handle;

      NVFBC_CREATE_CAPTURE_SESSION_PARAMS capture_params;
    };
  }  // namespace nvfbc
}  // namespace cuda

namespace platf {
  std::shared_ptr<display_t>
  nvfbc_display(mem_type_e hwdevice_type, const std::string &display_name, const video::config_t &config) {
    if (hwdevice_type != mem_type_e::cuda) {
      BOOST_LOG(error) << "Could not initialize nvfbc display with the given hw device type"sv;
      return nullptr;
    }

    auto display = std::make_shared<cuda::nvfbc::display_t>();

    if (display->init(display_name, config)) {
      return nullptr;
    }

    return display;
  }

  std::vector<std::string>
  nvfbc_display_names() {
    if (cuda::init() || cuda::nvfbc::init()) {
      return {};
    }

    std::vector<std::string> display_names;

    auto handle = cuda::nvfbc::handle_t::make();
    if (!handle) {
      return {};
    }

    auto status_params = handle->status();
    if (!status_params) {
      return {};
    }

    if (!status_params->bIsCapturePossible) {
      BOOST_LOG(error) << "NVidia driver doesn't support NvFBC screencasting"sv;
    }

    BOOST_LOG(info) << "Found ["sv << status_params->dwOutputNum << "] outputs"sv;
    BOOST_LOG(info) << "Virtual Desktop: "sv << status_params->screenSize.w << 'x' << status_params->screenSize.h;
    BOOST_LOG(info) << "XrandR: "sv << (status_params->bXRandRAvailable ? "available"sv : "unavailable"sv);

    for (auto x = 0; x < status_params->dwOutputNum; ++x) {
      auto &output = status_params->outputs[x];
      BOOST_LOG(info) << "-- Output --"sv;
      BOOST_LOG(debug) << "  ID: "sv << output.dwId;
      BOOST_LOG(debug) << "  Name: "sv << output.name;
      BOOST_LOG(info) << "  Resolution: "sv << output.trackedBox.w << 'x' << output.trackedBox.h;
      BOOST_LOG(info) << "  Offset: "sv << output.trackedBox.x << 'x' << output.trackedBox.y;
      display_names.emplace_back(std::to_string(x));
    }

    return display_names;
  }
}  // namespace platf
