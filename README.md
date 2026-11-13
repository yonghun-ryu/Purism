# Purism: In-TCU Dequantization for Quantized-Weight GEMM

Purism extends the tensor core unit (TCU) of the [Vortex](https://github.com/vortexgpgpu/vortex) v2.3 RISC-V GPGPU
so that quantized-weight GEMM runs without SIMT dequantization instructions.

- An **operand conversion unit (OCU)** on the TCU operand path unpacks and converts the weight (B) operand
  and subtracts its zero-point.
- A **partial-sum scaling unit (PSSU)** on the TCU output path applies group scales and accumulates in FP32.

```
operand collector ─▶ TCU dispatch ─▶ OCU (operand transformer) ─┬─▶ FP PE ─┐
                                                                 ├─▶ INT PE ─┼─▶ arbiter ─▶ PSSU ─▶ commit
                                                                 └─▶ bypass ─┘   (SETSCALE/READ/SETZP)
```

## Instructions (RISC-V custom-0, opcode `0x0B`)

| Instruction | funct7 | funct3 | Operands |
|---|---|---|---|
| `WMMA` | 0x02 | 0 | rs1 field = TCU type (FP16 / INT8), rd field = output type, **rs2 field = B source format** (0 = no conversion) |
| `WMMA.DRAIN` | 0x02 | 1/2 = one-scale bank 0/1, 3/4 = two-scale bank 0/1 | as `WMMA`; the last k-step result (group partial sum) goes to the PSSU |
| `PSSU.SETSCALE` | 0x03 | 0 | rd field = {bank, weight/activation, slot}, rs1 = F-reg with per-lane scales |
| `PSSU.READ` | 0x03 | 1 | rd = F-reg ← accumulator entry, then the entry is cleared; rs1 field = entry index |
| `OT.SETZP` | 0x03 | 2 | rs1 = 64-bit I-reg with per-lane zero-points (8 bits per unpacked B register) |

All of them issue to the TCU, so program order is kept inside the TCU path.

- **OCU**: INT4/UINT4 → INT8, FP8 E4M3, FP16; INT8/UINT8 → FP16; FP8 E4M3/E5M2 → FP16; FP4 E2M1 → FP8 E4M3, FP16.
  A packed B register is expanded over consecutive micro-ops, so the register file holds up to 4× less B data.
  Zero-points are per warp, per lane and per unpacked B register.
- **PSSU**: `Y ← Y + f(P) · Sw · (Sa | 1)` per group partial sum `P`, with an internal FP32 accumulator `Y`,
  INT32 → FP32 conversion for the INT8 path, and double-banked scales so that the next group's scales load
  while the current group drains.

## Source map

| Path | Content |
|---|---|
| `hw/rtl/tcu/VX_tcu_operand_transformer.sv` | OCU stage |
| `hw/rtl/tcu/VX_tcu_pssu.sv` | PSSU stage |
| `hw/rtl/tcu/VX_tcu_unit.sv`, `VX_tcu_uops.sv` | integration, micro-op expansion |
| `hw/rtl/core/VX_decode.sv`, `VX_uop_sequencer.sv` | instruction decode |
| `hw/rtl/ocu/` | format converters, zero-point subtraction (`tb_ocu_word.sv`, `tb_ot_formats.sv`) |
| `hw/rtl/pssu/` | PSSU core: FP32 multiply and round-to-nearest-even add (`tb_pssu_core_v2.sv`) |
| `tests/regression/qgemm` | W4A16 GEMM kernel (INT4 weights, FP16 activations, per-group scale and zero-point) |
| `tests/regression/purism_gemm` | GEMM kernel over the source formats (`-DPG_SRC=<format id>`, `-DPG_INT8` for INT8 output) |

## Build and run

Purism is enabled with `-DEXT_TCU_ENABLE -DEXT_PURISM_ENABLE` (the configuration below also uses
`-DEXT_D_DISABLE -DNUM_THREADS=32 -DNUM_WARPS=4 -DNUM_OPCS=1`). Set up the toolchain as described in
the upstream section below.

```bash
# RTL simulation (Verilator)
CONFIGS="-DEXT_TCU_ENABLE -DEXT_PURISM_ENABLE -DEXT_D_DISABLE -DNUM_THREADS=32 -DNUM_WARPS=4 -DNUM_OPCS=1 -DOPC_DISPATCH_PIPE=1 -DLMEM_DISABLE" \
  DESTDIR=<rt_dir> XLEN=64 make -C runtime/rtlsim all
make -C tests/regression/qgemm XLEN=64 CONFIGS="-DNUM_THREADS=32" VORTEX_RT_PATH=<rt_dir>
cd tests/regression/qgemm && LD_LIBRARY_PATH=<rt_dir> VORTEX_DRIVER=rtlsim ./qgemm -m 16 -n 32 -k 128 -g 128 -Z

# Xilinx Alveo U250 (single core, 100 MHz)
cd hw/syn/xilinx/xrt
export CONFIGS="-DEXT_TCU_ENABLE -DEXT_PURISM_ENABLE -DEXT_D_DISABLE -DNUM_THREADS=32 -DNUM_WARPS=4 -DNUM_OPCS=1 -DOPC_DISPATCH_PIPE=2 -DDISPATCH_ACCEL_PIPE=1"
NT32_SPREAD=1 NT32_BASE_SLR3=1 make NUM_CORES=1 PLATFORM=<u250 .xpfm> KERNEL_FREQ=100 XLEN=64 TARGET=hw PLACE_NO_BUFG=1
```

`-DLMEM_DISABLE` works around a Verilator 5.028 start-up crash in the local-memory crossbar; the test
kernels do not use local memory. Omitting `-DEXT_PURISM_ENABLE` builds the same GPU without Purism.

---

# Vortex GPGPU (upstream)

Vortex is a full-stack open-source RISC-V GPGPU. Vortex supports multiple **backend drivers**, including our C++ simulator (simx), an RTL simulator, and physical Xilinx and Altera FPGAs-- all controlled by a single driver script. The chosen driver determines the corresponding code invoked to run Vortex. Generally, developers will prototype their intended design in simx, before completing going forward with an RTL implementation. Alternatively, you can get up and running by selecting a driver of your choice and running a demo program.

## Website
Vortex news can be found on its [website](https://vortex.cc.gatech.edu/)

## Citation
```
@inproceedings{10.1145/3466752.3480128,
	author = {Tine, Blaise and Yalamarthy, Krishna Praveen and Elsabbagh, Fares and Hyesoon, Kim},
	title = {Vortex: Extending the RISC-V ISA for GPGPU and 3D-Graphics},
	year = {2021},
	isbn = {9781450385572},
	publisher = {Association for Computing Machinery},
	address = {New York, NY, USA},
	url = {https://doi.org/10.1145/3466752.3480128},
	doi = {10.1145/3466752.3480128},
	abstract = {The importance of open-source hardware and software has been increasing. However, despite GPUs being one of the more popular accelerators across various applications, there is very little open-source GPU infrastructure in the public domain. We argue that one of the reasons for the lack of open-source infrastructure for GPUs is rooted in the complexity of their ISA and software stacks. In this work, we first propose an ISA extension to RISC-V that supports GPGPUs and graphics. The main goal of the ISA extension proposal is to minimize the ISA changes so that the corresponding changes to the open-source ecosystem are also minimal, which makes for a sustainable development ecosystem. To demonstrate the feasibility of the minimally extended RISC-V ISA, we implemented the complete software and hardware stacks of Vortex on FPGA. Vortex is a PCIe-based soft GPU that supports OpenCL and OpenGL. Vortex can be used in a variety of applications, including machine learning, graph analytics, and graphics rendering. Vortex can scale up to 32 cores on an Altera Stratix 10 FPGA, delivering a peak performance of 25.6 GFlops at 200 Mhz.},
	booktitle = {MICRO-54: 54th Annual IEEE/ACM International Symposium on Microarchitecture},
	pages = {754–766},
	numpages = {13},
	keywords = {reconfigurable computing, memory systems., computer graphics},
	location = {Virtual Event, Greece},
	series = {MICRO '21}
}
```

## Specifications

- Support RISC-V RV32IMAF and RV64IMAFD

- Microarchitecture:
    - configurable number of cores, warps, and threads.
    - configurable number of ALU, FPU, LSU, and SFU units per core.
    - configurable pipeline issue width.
    - optional local memory, L1, L2, and L3 caches.
- Software:
    - OpenCL 1.2 Support.
- Supported FPGAs:
    - Altera Arria 10
    - Altera Stratix 10
    - Xilinx Alveo U50, U250, U280
    - Xilinx Versal VCK5000

## Directory structure

- `doc`: [Documentation](docs/index.md).
- `hw`: Hardware sources.
- `driver`: Host drivers repository.
- `runtime`: Kernel Runtime software.
- `sim`: Simulators repository.
- `tests`: Tests repository.
- `ci`: Continuous integration scripts.
- `miscs`: Miscellaneous resources.

## Quick Start
If you are interested in a stable release of Vortex, you can download the latest release [here](https://github.com/vortexgpgpu/vortex/releases/latest). Otherwise, you can pull the most recent, but (potentially) unstable version as shown below. The following steps demonstrate how to build and run Vortex with the default driver: SimX. If you are interested in a different backend, look [here](docs/simulation.md).

### Supported OS Platforms
- Ubuntu 22.04, 24.04
- Centos 7
### Toolchain Dependencies
The following dependencies will be fetched prebuilt by `toolchain_install.sh`.
- [POCL](http://portablecl.org/)
- [LLVM](https://llvm.org/)
- [RISCV-GNU-TOOLCHAIN](https://github.com/riscv-collab/riscv-gnu-toolchain)
- [Verilator](https://www.veripool.org/verilator)
- [cvfpu](https://github.com/openhwgroup/cvfpu.git)
- [SoftFloat](https://github.com/ucb-bar/berkeley-softfloat-3.git)
- [Ramulator](https://github.com/CMU-SAFARI/ramulator.git)
- [Yosys](https://github.com/YosysHQ/yosys)
- [Sv2v](https://github.com/zachjs/sv2v)
### Install Vortex codebase
```sh
	git clone --depth=1 --recursive https://github.com/vortexgpgpu/vortex.git
	cd vortex
```
### Install system dependencies
```sh
# ensure dependent libraries are present
sudo ./ci/install_dependencies.sh
```
### Configure your build folder
```sh
    mkdir build
    cd build
    # for 32bit
    ../configure --xlen=32 --tooldir=$HOME/tools
    # for 64bit
    ../configure --xlen=64 --tooldir=$HOME/tools
```
### Install prebuilt toolchain
```sh
   ./ci/toolchain_install.sh --all
```
### set environment variables
```sh
    # should always run before using the toolchain!
    source ./ci/toolchain_env.sh
```
### Building Vortex
```sh
make -s
```
### Quick demo running vecadd OpenCL kernel on 2 cores
```sh
./ci/blackbox.sh --cores=2 --app=vecadd
```

### Common Developer Tips
- Installing Vortex kernel and runtime libraries to use with external tools requires passing --prefix=<install-path> to the configure script.
```sh
../configure --xlen=32 --tooldir=$HOME/tools --prefix=<install-path>
make -s
make install
```
- Building Vortex 64-bit requires setting --xlen=64 configure option.
```sh
../configure --xlen=64 --tooldir=$HOME/tools
```
- Sourcing "./ci/toolchain_env.sh" is required everytime you start a new terminal. we recommend adding "source <build-path>/ci/toolchain_env.sh" to your ~/.bashrc file to automate the process at login.
```sh
echo "source <build-path>/ci/toolchain_env.sh" >> ~/.bashrc
```
- Making changes to Makefiles in your source tree or adding new folders will require executing the "configure" script again without any options to get changes propagated to your build folder.
```sh
../configure
```
- To debug the GPU, the simulation can generate a runtime trace for analysis. See /docs/debugging.md for more information.
```sh
./ci/blackbox.sh --app=demo --debug=3
```
- For additional information, check out the [documentation](docs/index.md)
