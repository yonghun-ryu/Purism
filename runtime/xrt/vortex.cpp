// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <common.h>

#ifdef SCOPE
#include "scope.h"
#endif

// XRT includes
#ifdef XRTSIM
#include <xrt_c.h>
#else
#include "experimental/xrt_bo.h"
#include "experimental/xrt_device.h"
#include "experimental/xrt_error.h"
#include "experimental/xrt_ip.h"
#include "experimental/xrt_kernel.h"
#include "experimental/xrt_xclbin.h"
#endif

#include <limits>
#include <stdarg.h>
#include <string>
#include <unordered_map>
#include <util.h>
#include <vector>

using namespace vortex;

#ifndef XRTSIM
#define CPP_API
#endif

// #define BANK_INTERLEAVE

#define MMIO_CTL_ADDR 0x00
#define MMIO_DEV_ADDR 0x10
#define MMIO_ISA_ADDR 0x18
#define MMIO_DCR_ADDR 0x20
#define MMIO_SCP_ADDR 0x28
#define MMIO_MEM_ADDR 0x30

#define CTL_AP_START (1 << 0)
#define CTL_AP_DONE (1 << 1)
#define CTL_AP_IDLE (1 << 2)
#define CTL_AP_READY (1 << 3)
#define CTL_AP_RESET (1 << 4)
#define CTL_AP_RESTART (1 << 7)

#ifdef CPP_API

typedef xrt::device xrt_device_t;
typedef xrt::ip xrt_kernel_t;
typedef xrt::bo xrt_buffer_t;

#else

typedef xrtDeviceHandle xrt_device_t;
typedef xrtKernelHandle xrt_kernel_t;
typedef xrtBufferHandle xrt_buffer_t;

#endif

#define DEFAULT_DEVICE_INDEX 0

#define DEFAULT_XCLBIN_PATH "vortex_afu.xclbin"

#define KERNEL_NAME "vortex_afu"

#define CHECK_HANDLE(handle, _expr, _cleanup)                                  \
  auto handle = _expr;                                                         \
  if (handle == nullptr) {                                                     \
    printf("[VXDRV] Error: '%s' returned NULL!\n", #_expr);                    \
    _cleanup                                                                   \
  }

#ifndef CPP_API
static void dump_xrt_error(xrtDeviceHandle xrtDevice, xrtErrorCode err) {
  size_t len = 0;
  xrtErrorGetString(xrtDevice, err, nullptr, 0, &len);
  std::vector<char> buf(len);
  xrtErrorGetString(xrtDevice, err, buf.data(), buf.size(), nullptr);
  printf("[VXDRV] detail: %s!\n", buf.data());
}
#endif

///////////////////////////////////////////////////////////////////////////////

class vx_device {
public:
  vx_device()
    : global_mem_(ALLOC_BASE_ADDR,
                  GLOBAL_MEM_SIZE - ALLOC_BASE_ADDR,
                  RAM_PAGE_SIZE,
                  CACHE_BLOCK_SIZE)
  #ifndef CPP_API
    , xrtDevice_(nullptr)
    , xrtKernel_(nullptr)
  #endif
    , xrt_mem_group_(0)
  {}

  ~vx_device() {
  #ifdef SCOPE
    vx_scope_stop(this);
  #endif
  #ifndef CPP_API
    for (auto &entry : xrtBuffers_) {
    #ifdef BANK_INTERLEAVE
      xrtBOFree(entry);
    #else
      xrtBOFree(entry.second.xrtBuffer);
    #endif
    }
    if (xrtKernel_) {
      xrtKernelClose(xrtKernel_);
    }
    if (xrtDevice_) {
      xrtDeviceClose(xrtDevice_);
    }
  #endif
  }

  int init() {
    int device_index = DEFAULT_DEVICE_INDEX;
    const char *device_index_s = getenv("XRT_DEVICE_INDEX");
    if (device_index_s != nullptr) {
      device_index = atoi(device_index_s);
    }

    const char *xlbin_path_s = getenv("XRT_XCLBIN_PATH");
    if (xlbin_path_s == nullptr) {
      xlbin_path_s = DEFAULT_XCLBIN_PATH;
    }

  #ifdef CPP_API

    auto xrtDevice = xrt::device(device_index);
    auto uuid = xrtDevice.load_xclbin(xlbin_path_s);
    auto xrtKernel = xrt::ip(xrtDevice, uuid, KERNEL_NAME);
    auto xclbin = xrt::xclbin(xlbin_path_s);
    auto device_name = xrtDevice.get_info<xrt::info::device::name>();

    // Find the first used DDR memory group in the xclbin
    xrt_mem_group_ = 0;
    for (auto& mem : xclbin.get_mems()) {
      if (mem.get_used()) {
        xrt_mem_group_ = mem.get_index();
        break;
      }
    }
    printf("info: using XRT memory group %u\n", xrt_mem_group_);

  #else

    CHECK_HANDLE(xrtDevice, xrtDeviceOpen(device_index), {
      return -1;
    });

  #ifndef XRTSIM
    CHECK_ERR(xrtDeviceLoadXclbinFile(xrtDevice, xlbin_path_s), {
      dump_xrt_error(xrtDevice, err);
      xrtDeviceClose(xrtDevice);
      return err;
    });

    xuid_t uuid;
    CHECK_ERR(xrtDeviceGetXclbinUUID(xrtDevice, uuid), {
      dump_xrt_error(xrtDevice, err);
      xrtDeviceClose(xrtDevice);
      return err;
    });

    CHECK_HANDLE(xrtKernel, xrtPLKernelOpenExclusive(xrtDevice, uuid, KERNEL_NAME), {
      xrtDeviceClose(xrtDevice);
      return -1;
    });
  #else
    xrtKernelHandle xrtKernel = xrtDevice;
  #endif

    // get device name
    int device_name_size;
    xrtXclbinGetXSAName(xrtDevice, nullptr, 0, &device_name_size);
    std::vector<char> sz_device_name(device_name_size);
    xrtXclbinGetXSAName(xrtDevice, sz_device_name.data(), device_name_size, nullptr);
    std::string device_name(sz_device_name.data(), device_name_size);

  #endif

    xrtDevice_ = xrtDevice;
    xrtKernel_ = xrtKernel;

    CHECK_ERR(this->write_register(MMIO_CTL_ADDR, CTL_AP_RESET), {
      return err;
    });

    CHECK_ERR(this->read_register(MMIO_DEV_ADDR, (uint32_t *)&dev_caps_), {
      return err;
    });

    CHECK_ERR(this->read_register(MMIO_DEV_ADDR + 4, (uint32_t *)&dev_caps_ + 1), {
      return err;
    });

    CHECK_ERR(this->read_register(MMIO_ISA_ADDR, (uint32_t *)&isa_caps_), {
      return err;
    });

    CHECK_ERR(this->read_register(MMIO_ISA_ADDR + 4, (uint32_t *)&isa_caps_ + 1), {
      return err;
    });

    uint64_t num_banks;
    this->get_caps(VX_CAPS_NUM_MEM_BANKS, &num_banks);
    lg2_num_banks_ = log2ceil(num_banks);

    uint64_t bank_size;
    this->get_caps(VX_CAPS_MEM_BANK_SIZE, &bank_size);
    lg2_bank_size_ = log2ceil(bank_size);

    global_mem_size_ = num_banks * bank_size;

    printf("info: device name=%s, memory_capacity=0x%lx bytes, memory_banks=%ld.\n", device_name.c_str(), global_mem_size_, num_banks);

    // Tier-0 (>8GB single-DDR[0]): the kernel stack (grows down from STACK_BASE,
    // in DRAM) and the LMEM decode window (at LMEM_BASE, routed on-chip) sit at
    // ~8GB, now inside the enlarged allocatable range. Reserve that band so the
    // first-fit allocator never hands it out to a data buffer (which would be
    // corrupted by stack writes / silently shadowed to on-chip LMEM). No-op on builds where
    // the band lies at/above the cap (e.g. the legacy 8GB layout).
    {
      uint64_t nt = 0, nw = 0, nc = 0;
      this->get_caps(VX_CAPS_NUM_THREADS, &nt);
      this->get_caps(VX_CAPS_NUM_WARPS,   &nw);
      this->get_caps(VX_CAPS_NUM_CORES,   &nc);
      uint64_t harts    = (nt ? nt : 1) * (nw ? nw : 1) * (nc ? nc : 1);
      uint64_t stack_sz = harts << STACK_LOG2_SIZE;
      uint64_t band_lo  = (uint64_t)STACK_BASE_ADDR - stack_sz;
      uint64_t band_hi  = (uint64_t)LMEM_BASE_ADDR + (1ull << LMEM_LOG_SIZE);
      if (band_lo >= (uint64_t)ALLOC_BASE_ADDR && band_hi <= (uint64_t)GLOBAL_MEM_SIZE) {
        if (0 == global_mem_.reserve(band_lo, band_hi - band_lo)) {
          printf("info: reserved stack/LMEM band [0x%lx, 0x%lx) for %lu harts\n",
                 band_lo, band_hi, harts);
        }
      }
    }

  #ifdef BANK_INTERLEAVE
    xrtBuffers_.reserve(1);
    {
      uint32_t i = 0;
    #ifdef CPP_API
      xrtBuffers_.emplace_back(xrtDevice_, (uint64_t)GLOBAL_MEM_SIZE, xrt::bo::flags::normal, xrt_mem_group_);
    #else
      CHECK_HANDLE(xrtBuffer, xrtBOAlloc(xrtDevice_, (uint64_t)GLOBAL_MEM_SIZE, XRT_BO_FLAGS_NONE, xrt_mem_group_), {
         return -1;
      });
      xrtBuffers_.push_back(xrtBuffer);
    #endif
      printf("*** allocated mem_group=%u, size=%lu\n", xrt_mem_group_, (uint64_t)GLOBAL_MEM_SIZE);
    }
  #endif

  #ifdef SCOPE
    {
      scope_callback_t callback;
      callback.registerWrite = [](vx_device_h hdevice, uint64_t value) -> int {
        auto device = (vx_device *)hdevice;
        uint32_t value_lo = (uint32_t)(value);
        uint32_t value_hi = (uint32_t)(value >> 32);
        CHECK_ERR(device->write_register(MMIO_SCP_ADDR, value_lo), {
          return err;
        });
        CHECK_ERR(device->write_register(MMIO_SCP_ADDR + 4, value_hi), {
          return err;
        });
        return 0;
      };
      callback.registerRead = [](vx_device_h hdevice, uint64_t *value) -> int {
        auto device = (vx_device *)hdevice;
        uint32_t value_lo, value_hi;
        CHECK_ERR(device->read_register(MMIO_SCP_ADDR, &value_lo), {
          return err;
        });
        CHECK_ERR(device->read_register(MMIO_SCP_ADDR + 4, &value_hi), {
          return err;
        });
        *value = (((uint64_t)value_hi) << 32) | value_lo;
        return 0;
      };
      CHECK_ERR(vx_scope_start(&callback, this, -1, -1), {
        return err;
      });
    }
  #endif

  #ifdef CHIPSCOPE
    std::cout << "\nPress ENTER to continue after setting up ILA trigger..." << std::endl;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  #endif

    return 0;
  }

  int get_caps(uint32_t caps_id, uint64_t *value) {
    uint64_t _value;

    switch (caps_id) {
    case VX_CAPS_VERSION:
      _value = (dev_caps_ >> 0) & 0xff;
      break;
    case VX_CAPS_NUM_THREADS:
      _value = (dev_caps_ >> 8) & 0xff;
      break;
    case VX_CAPS_NUM_WARPS:
      _value = (dev_caps_ >> 16) & 0xff;
      break;
    case VX_CAPS_NUM_CORES:
      _value = (dev_caps_ >> 24) & 0xffff;
      break;
    case VX_CAPS_CACHE_LINE_SIZE:
      _value = CACHE_BLOCK_SIZE;
      break;
    case VX_CAPS_GLOBAL_MEM_SIZE:
      _value = global_mem_size_;
      break;
    case VX_CAPS_LOCAL_MEM_SIZE:
      _value = 1ull << ((dev_caps_ >> 40) & 0xff);
      break;
    case VX_CAPS_ISA_FLAGS:
      _value = isa_caps_;
      break;
    case VX_CAPS_NUM_MEM_BANKS:
      _value = 1 << ((dev_caps_ >> 48) & 0x7);
      break;
    case VX_CAPS_MEM_BANK_SIZE:
      _value = 1ull << (20 + ((dev_caps_ >> 51) & 0x1f));
      break;
    default:
      fprintf(stderr, "[VXDRV] Error: invalid caps id: %d\n", caps_id);
      std::abort();
      return -1;
    }

    *value = _value;

    return 0;
  }

  int mem_alloc(uint64_t size, int flags, uint64_t *dev_addr) {
    uint64_t asize = aligned_size(size, CACHE_BLOCK_SIZE);
    uint64_t addr;
    CHECK_ERR(global_mem_.allocate(asize, &addr), {
      return err;
    });
  #ifndef BANK_INTERLEAVE
    uint32_t bank_id;
    CHECK_ERR(this->get_bank_info(addr, &bank_id, nullptr), {
      global_mem_.release(addr);
      return err;
    });
    CHECK_ERR(get_buffer(bank_id, nullptr), {
      global_mem_.release(addr);
      return err;
    });
  #endif
    CHECK_ERR(this->mem_access(addr, size, flags), {
      global_mem_.release(addr);
      return err;
    });
    *dev_addr = addr;
    return 0;
  }

  int mem_reserve(uint64_t dev_addr, uint64_t size, int flags) {
    CHECK_ERR(global_mem_.reserve(dev_addr, size), {
      return err;
    });
  #ifndef BANK_INTERLEAVE
    uint32_t bank_id;
    CHECK_ERR(this->get_bank_info(dev_addr, &bank_id, nullptr), {
      global_mem_.release(dev_addr);
      return err;
    });
    CHECK_ERR(get_buffer(bank_id, nullptr), {
      global_mem_.release(dev_addr);
      return err;
    });
  #endif
    CHECK_ERR(this->mem_access(dev_addr, size, flags), {
      global_mem_.release(dev_addr);
      return err;
    });
    return 0;
  }

  int mem_free(uint64_t dev_addr) {
    CHECK_ERR(global_mem_.release(dev_addr), {
      return err;
    });
  #ifdef BANK_INTERLEAVE
    if (0 == global_mem_.allocated()) {
    #ifndef CPP_API
      for (auto &entry : xrtBuffers_) {
        xrtBOFree(entry);
      }
    #endif
      xrtBuffers_.clear();
    }
  #else
    uint32_t bank_id;
    CHECK_ERR(this->get_bank_info(dev_addr, &bank_id, nullptr), {
      return err;
    });
    auto it = xrtBuffers_.find(bank_id);
    if (it != xrtBuffers_.end()) {
      auto count = --it->second.count;
      if (0 == count) {
        printf("freeing bank%d...\n", bank_id);
      #ifndef CPP_API
        xrtBOFree(it->second.xrtBuffer);
      #endif
        xrtBuffers_.erase(it);
      }
    } else {
      fprintf(stderr, "[VXDRV] Error: invalid device memory address: 0x%lx\n",
              dev_addr);
      return -1;
    }
  #endif
    return 0;
  }

  int mem_access(uint64_t /*dev_addr*/, uint64_t /*size*/, int /*flags*/) {
    return 0;
  }

  int mem_info(uint64_t *mem_free, uint64_t *mem_used) const {
    if (mem_free)
      *mem_free = global_mem_.free();
    if (mem_used)
      *mem_used = global_mem_.allocated();
    return 0;
  }

  int write_register(uint32_t addr, uint32_t value) {
  #ifdef CPP_API
    xrtKernel_.write_register(addr, value);
  #else
    CHECK_ERR(xrtKernelWriteRegister(xrtKernel_, addr, value), {
      dump_xrt_error(xrtDevice_, err);
      return err;
    });
  #endif
    return 0;
  }

  int read_register(uint32_t addr, uint32_t *value) {
  #ifdef CPP_API
    *value = xrtKernel_.read_register(addr);
  #else
    CHECK_ERR(xrtKernelReadRegister(xrtKernel_, addr, value), {
      dump_xrt_error(xrtDevice_, err);
      return err;
    });
  #endif
    return 0;
  }

  int copy(uint64_t dest_addr, uint64_t src_addr, uint64_t size) {
    if (dest_addr == src_addr) {
      return 0;
    }

    // bound checking
    if (dest_addr + size > global_mem_size_ ||
        src_addr + size > global_mem_size_)
      return -1;

    uint64_t offset = 0;
    while (offset < size) {
      uint64_t curr_src = src_addr + offset;
      uint64_t curr_dest = dest_addr + offset;

      uint64_t src_rem = CACHE_BLOCK_SIZE - (curr_src % CACHE_BLOCK_SIZE);
      uint64_t dest_rem = CACHE_BLOCK_SIZE - (curr_dest % CACHE_BLOCK_SIZE);

      uint64_t chunk_size = (src_rem < dest_rem) ? src_rem : dest_rem;
      if (chunk_size > size - offset) {
        chunk_size = size - offset;
      }

      uint32_t src_bo_idx, dst_bo_idx;
      uint64_t src_bo_off, dst_bo_off;
      xrt_buffer_t src_buf, dst_buf;

      CHECK_ERR(this->get_bank_info(curr_src, &src_bo_idx, &src_bo_off), {
        return err;
      });
#ifdef BANK_INTERLEAVE
      src_bo_off += (curr_src % CACHE_BLOCK_SIZE);
#endif

      CHECK_ERR(this->get_buffer(src_bo_idx, &src_buf), {
        return err;
      });

      CHECK_ERR(this->get_bank_info(curr_dest, &dst_bo_idx, &dst_bo_off), {
        return err;
      });
#ifdef BANK_INTERLEAVE
      dst_bo_off += (curr_dest % CACHE_BLOCK_SIZE);
#endif

      CHECK_ERR(this->get_buffer(dst_bo_idx, &dst_buf), {
        return err;
      });

#ifdef CPP_API
      dst_buf.copy(src_buf, chunk_size, src_bo_off, dst_bo_off);
#else
      CHECK_ERR(xrtBOCopy(dst_buf, src_buf, chunk_size, src_bo_off, dst_bo_off), {
        dump_xrt_error(xrtDevice_, err);
        return err;
      });
#endif

      offset += chunk_size;
    }

    return 0;
  }

  int upload(uint64_t dev_addr, const void *src, uint64_t size) {
    auto host_ptr = (const uint8_t *)src;

    // check alignment
    if (!is_aligned(dev_addr, CACHE_BLOCK_SIZE))
      return -1;

    auto asize = aligned_size(size, CACHE_BLOCK_SIZE);

    // bound checking
    if (dev_addr + asize > global_mem_size_)
      return -1;

    for (uint64_t end = dev_addr + asize; dev_addr < end;
         dev_addr += CACHE_BLOCK_SIZE, host_ptr += CACHE_BLOCK_SIZE) {
    #ifdef BANK_INTERLEAVE
      asize = CACHE_BLOCK_SIZE;
    #else
      end = 0;
    #endif
      uint32_t bo_index;
      uint64_t bo_offset;
      xrt_buffer_t xrtBuffer;
      CHECK_ERR(this->get_bank_info(dev_addr, &bo_index, &bo_offset), {
        return err;
      });
      CHECK_ERR(this->get_buffer(bo_index, &xrtBuffer), {
        return err;
      });
    #ifdef CPP_API
      // xrt::bo::write/sync throw on failure; convert to an error return so a
      // failed transfer never aborts the whole process (matches the C-API branch).
      try {
        xrtBuffer.write(host_ptr, size, bo_offset);
        xrtBuffer.sync(XCL_BO_SYNC_BO_TO_DEVICE, size, bo_offset);
      } catch (const std::exception& e) {
        printf("[VXDRV] Error: BO write/sync-to-device failed (bo_offset=0x%lx size=0x%lx): %s\n",
               bo_offset, size, e.what());
        return -1;
      }
    #else
      CHECK_ERR(xrtBOWrite(xrtBuffer, host_ptr, size, bo_offset), {
        dump_xrt_error(xrtDevice_, err);
        return err;
      });
      CHECK_ERR(xrtBOSync(xrtBuffer, XCL_BO_SYNC_BO_TO_DEVICE, size, bo_offset), {
        dump_xrt_error(xrtDevice_, err);
        return err;
      });
#endif
    }
    return 0;
  }

  int download(void *dest, uint64_t dev_addr, uint64_t size) {
    auto host_ptr = (uint8_t *)dest;

    // check alignment
    if (!is_aligned(dev_addr, CACHE_BLOCK_SIZE))
      return -1;

    auto asize = aligned_size(size, CACHE_BLOCK_SIZE);

    // bound checking
    if (dev_addr + asize > global_mem_size_)
      return -1;

    for (uint64_t end = dev_addr + asize; dev_addr < end;
         dev_addr += CACHE_BLOCK_SIZE, host_ptr += CACHE_BLOCK_SIZE) {
    #ifdef BANK_INTERLEAVE
      asize = CACHE_BLOCK_SIZE;
    #else
      end = 0;
    #endif
      uint32_t bo_index;
      uint64_t bo_offset;
      xrt_buffer_t xrtBuffer;
      CHECK_ERR(this->get_bank_info(dev_addr, &bo_index, &bo_offset), {
        return err;
      });
      CHECK_ERR(this->get_buffer(bo_index, &xrtBuffer), {
        return err;
      });
    #ifdef CPP_API
      // xrt::bo::sync/read throw on failure; convert to an error return so a
      // failed transfer never aborts the whole process (matches the C-API branch).
      // Notably this download() serves the MPM/perf read from vx_dev_close ->
      // vx_dump_perf: when all user buffers were freed, get_buffer lazily
      // re-allocates a pristine bank BO whose FROM_DEVICE sync XRT rejects
      // ("Invalid argument"). Returning -1 lets vx_dump_perf skip gracefully.
      try {
        xrtBuffer.sync(XCL_BO_SYNC_BO_FROM_DEVICE, size, bo_offset);
        xrtBuffer.read(host_ptr, size, bo_offset);
      } catch (const std::exception& e) {
        printf("[VXDRV] Error: BO sync-from-device/read failed (bo_offset=0x%lx size=0x%lx): %s\n",
               bo_offset, size, e.what());
        return -1;
      }
    #else
      CHECK_ERR(xrtBOSync(xrtBuffer, XCL_BO_SYNC_BO_FROM_DEVICE, size, bo_offset), {
        dump_xrt_error(xrtDevice_, err);
        return err;
      });
      CHECK_ERR(xrtBORead(xrtBuffer, host_ptr, size, bo_offset), {
        dump_xrt_error(xrtDevice_, err);
        return err;
      });
    #endif
    }
    return 0;
  }

  int start(uint64_t krnl_addr, uint64_t args_addr) {
    printf("start: krnl_addr=0x%lx, args_addr=0x%lx\n", krnl_addr, args_addr);
    fflush(stdout);
    // set kernel info
    CHECK_ERR(this->dcr_write(VX_DCR_BASE_STARTUP_ADDR0, krnl_addr & 0xffffffff), {
      return err;
    });
    CHECK_ERR(this->dcr_write(VX_DCR_BASE_STARTUP_ADDR1, krnl_addr >> 32), {
      return err;
    });
    CHECK_ERR(this->dcr_write(VX_DCR_BASE_STARTUP_ARG0, args_addr & 0xffffffff), {
      return err;
    });
    CHECK_ERR(this->dcr_write(VX_DCR_BASE_STARTUP_ARG1, args_addr >> 32), {
      return err;
    });

    // Wait for any stale AP_DONE from a previous run to clear before writing AP_START.
    // If AP_DONE=1 is present when AP_START is written, the Vortex AFU interprets it
    // as "acknowledge done → go idle" rather than starting a new execution.
    {
      uint32_t status = 0;
      for (int i = 0; i < 10; ++i) {
        this->read_register(MMIO_CTL_ADDR, &status);
        if (!(status & CTL_AP_DONE))
          break;
        printf("start: waiting for stale AP_DONE to clear (AP_CTL=0x%02x)...\n", status);
        fflush(stdout);
        struct timespec ts = {0, 1000000};  // 1 ms
        nanosleep(&ts, nullptr);
      }
      if (status & CTL_AP_DONE) {
        // Force-clear by writing AP_RESTART which acknowledges done without starting
        this->write_register(MMIO_CTL_ADDR, CTL_AP_RESTART);
        struct timespec ts2 = {0, 1000000};
        nanosleep(&ts2, nullptr);
      }
    }

    // start execution
    printf("start: writing AP_START=0x%x to MMIO_CTL_ADDR=0x%x\n", CTL_AP_START, MMIO_CTL_ADDR);
    CHECK_ERR(this->write_register(MMIO_CTL_ADDR, CTL_AP_START), {
      return err;
    });
    printf("start: AP_START written\n");
    {
      uint32_t status = 0;
      this->read_register(MMIO_CTL_ADDR, &status);
      printf("start: AP_CTL status after AP_START = 0x%x (idle=%d done=%d start=%d)\n",
             status, (status>>2)&1, (status>>1)&1, status&1);
    }

    // clear mpm cache
    mpm_cache_.clear();

    return 0;
  }

  int ready_wait(uint64_t timeout) {
    // Poll with exponential backoff (1ms -> 1000ms cap) instead of a flat
    // 1-second sleep. Most kernel launches in practice complete in well
    // under a second, so a fixed 1s poll granularity was adding up to
    // ~1s of pure polling latency per vx_start()/vx_ready_wait() pair --
    // dominant cost for workloads issuing many small launches (e.g.
    // an LLM decode with ~90 launches/token: ~90s/token of this
    // was polling latency, not compute or PCIe transfer time). The 1000ms
    // cap preserves the original steady-state poll rate (and printf
    // volume) for genuinely long-running kernels.
    uint64_t sleep_time_ms = 1;
    uint64_t elapsed_ms = 0;

    for (;;) {
      uint32_t status = 0;
      try {
        CHECK_ERR(this->read_register(MMIO_CTL_ADDR, &status), {
          return err;
        });
      } catch (std::exception& e) {
        printf("ready_wait: read_register threw: %s\n", e.what()); fflush(stdout);
        throw;
      }
      bool is_done = (status & CTL_AP_DONE) == CTL_AP_DONE;
      if (is_done)
        break;
      if (elapsed_ms >= timeout) {
        printf("ready_wait: TIMEOUT after %lums\n", elapsed_ms);
        return -1;
      }
      struct timespec sleep_time;
      sleep_time.tv_sec = sleep_time_ms / 1000;
      sleep_time.tv_nsec = (sleep_time_ms % 1000) * 1000000;
      nanosleep(&sleep_time, nullptr);
      elapsed_ms += sleep_time_ms;
      timeout -= sleep_time_ms;
      sleep_time_ms = (sleep_time_ms < 500) ? (sleep_time_ms * 2) : 1000;
    };

    return 0;
  }

  int dcr_write(uint32_t addr, uint32_t value) {
    CHECK_ERR(this->write_register(MMIO_DCR_ADDR, addr), {
      return err;
    });
    CHECK_ERR(this->write_register(MMIO_DCR_ADDR + 4, value), {
      return err;
    });
    dcrs_.write(addr, value);
    return 0;
  }

  int dcr_read(uint32_t addr, uint32_t *value) const {
    return dcrs_.read(addr, value);
  }

  int mpm_query(uint32_t addr, uint32_t core_id, uint64_t *value) {
    uint32_t offset = addr - VX_CSR_MPM_BASE;
    if (offset > 31)
      return -1;
    if (mpm_cache_.count(core_id) == 0) {
      uint64_t mpm_mem_addr = IO_MPM_ADDR + core_id * 32 * sizeof(uint64_t);
      CHECK_ERR(this->download(mpm_cache_[core_id].data(), mpm_mem_addr, 32 * sizeof(uint64_t)), {
        return err;
      });
    }
    *value = mpm_cache_.at(core_id).at(offset);
    return 0;
  }

private:

  MemoryAllocator global_mem_;
  xrt_device_t xrtDevice_;
  xrt_kernel_t xrtKernel_;
  uint64_t dev_caps_;
  uint64_t isa_caps_;
  uint32_t xrt_mem_group_;
  uint64_t global_mem_size_;
  DeviceConfig dcrs_;
  std::unordered_map<uint32_t, std::array<uint64_t, 32>> mpm_cache_;
  uint32_t lg2_num_banks_;
  uint32_t lg2_bank_size_;

#ifdef BANK_INTERLEAVE

  std::vector<xrt_buffer_t> xrtBuffers_;

  int get_bank_info(uint64_t addr, uint32_t *pIdx, uint64_t *pOff) {
    uint32_t num_banks = 1 << lg2_num_banks_;
    uint64_t block_addr = addr / CACHE_BLOCK_SIZE;
    uint32_t index = block_addr & (num_banks - 1);
    uint64_t offset = (block_addr >> lg2_num_banks_) * CACHE_BLOCK_SIZE;
    if (pIdx) {
      *pIdx = index;
    }
    if (pOff) {
      *pOff = offset;
    }
    //printf("get_bank_info(addr=0x%lx, bank=%d, offset=0x%lx\n", addr, index, offset);
    return 0;
  }

  int get_buffer(uint32_t bank_id, xrt_buffer_t *pBuf) {
    if (pBuf) {
      *pBuf = xrtBuffers_.at(bank_id);
    }
    return 0;
  }

#else

  struct buf_cnt_t {
    xrt_buffer_t xrtBuffer;
    uint32_t count;
  };

  std::unordered_map<uint32_t, buf_cnt_t> xrtBuffers_;

  int get_bank_info(uint64_t addr, uint32_t *pIdx, uint64_t *pOff) {
    uint32_t num_banks = 1 << lg2_num_banks_;
    uint64_t bank_size = 1ull << lg2_bank_size_;
    uint32_t index = addr >> lg2_bank_size_;
    uint64_t offset = addr & (bank_size - 1);
    if (index > num_banks) {
      fprintf(stderr, "[VXDRV] Error: address out of range: 0x%lx\n", addr);
      return -1;
    }
    if (pIdx) {
      *pIdx = index;
    }
    if (pOff) {
      *pOff = offset;
    }
    //printf("get_bank_info(addr=0x%lx, bank=%d, offset=0x%lx\n", addr, index, offset);
    return 0;
  }

  int get_buffer(uint32_t bank_id, xrt_buffer_t *pBuf) {
    auto it = xrtBuffers_.find(bank_id);
    if (it != xrtBuffers_.end()) {
      if (pBuf) {
        *pBuf = it->second.xrtBuffer;
      } else {
        printf("reusing bank%d...\n", bank_id);
        ++it->second.count;
      }
    } else {
      printf("allocating bank%d (mem_group=%u)...\n", bank_id, xrt_mem_group_);
      // Cap BO size to the SW allocator range. Physical DDR bank may be 16GB
      // but all virtual addresses are < GLOBAL_MEM_SIZE, so a GLOBAL_MEM_SIZE
      // BO covers the entire address range we ever use.
      uint64_t bank_size = std::min((uint64_t)(1ull << lg2_bank_size_), (uint64_t)GLOBAL_MEM_SIZE);
    #ifdef CPP_API
      xrt::bo xrtBuffer(xrtDevice_, bank_size, xrt::bo::flags::normal, xrt_mem_group_);
      printf("  BO allocated: phys_addr=0x%lx, size=0x%lx\n", xrtBuffer.address(), bank_size);
    #else
      CHECK_HANDLE(xrtBuffer, xrtBOAlloc(xrtDevice_, bank_size, XRT_BO_FLAGS_NONE, xrt_mem_group_), {
        return -1;
      });
    #endif
      xrtBuffers_.insert({bank_id, {xrtBuffer, 1}});
      if (pBuf) {
        *pBuf = xrtBuffer;
      }
    }
    return 0;
  }

#endif
};

#include <callbacks.inc>