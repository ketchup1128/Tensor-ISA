# RISC-V TensorLoad Extension Specification

Version 1.0, December 2024: This document is in development.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Implementation-defined Constant Parameters](#2-implementation-defined-constant-parameters)
3. [Programmer's Model](#3-programmers-model)
   - 3.1. [TensorLoad Registers](#31-tensorload-registers)
   - 3.2. [Control and Status Registers](#32-control-and-status-registers)
4. [Instructions](#4-instructions)
   - 4.1. [Instruction Formats](#41-instruction-formats)
   - 4.2. [Tensor Transpose Instructions](#42-tensor-transpose-instructions)
   - 4.3. [Tensor Concatenation Instructions](#43-tensor-concatenation-instructions)
   - 4.4. [Arithmetic Instructions](#44-arithmetic-instructions)
   - 4.5. [Memory Access Instructions](#45-memory-access-instructions)
5. [Standard TensorLoad Extensions](#5-standard-tensorload-extensions)
6. [TensorLoad Instruction Listing](#6-tensorload-instruction-listing)

---

## 1. Introduction

The TensorLoad extension provides specialized instructions for efficient tensor operations on RISC-V processors. This extension introduces a dedicated register file optimized for tensor data manipulation, including transpose, concatenation, arithmetic, and conditional memory access operations.

The TensorLoad extension is designed to accelerate machine learning workloads, signal processing, and other tensor-intensive applications by providing hardware-accelerated tensor operations with minimal software overhead.

### 1.1. Extension Dependencies

The TensorLoad extension depends on:
- RV32I or RV64I base integer instruction set
- Zicsr extension for CSR access

### 1.2. Extension Naming

The TensorLoad extension uses the naming convention `Ztl` for the base extension, with additional sub-extensions:
- `Ztl`: Base TensorLoad extension
- `Ztlx`: TensorLoad transpose operations
- `Ztlc`: TensorLoad concatenation operations
- `Ztla`: TensorLoad arithmetic operations
- `Ztlm`: TensorLoad memory operations

---

## 2. Implementation-defined Constant Parameters

The following parameters are implementation-defined:

| Parameter | Description | Constraints |
|-----------|-------------|-------------|
| TLREG_COUNT | Number of TensorLoad registers | Must be 32 |
| TLREG_SIZE | Size of each TensorLoad register in bytes | Must be 1024 |
| ELEMENT_SIZE | Size of each tensor element in bits | Must be 8 |
| MAX_TENSOR_DIM | Maximum tensor dimensions supported | Must be ≥ 4 |

---

## 3. Programmer's Model

### 3.1. TensorLoad Registers

The TensorLoad extension provides 32 dedicated tensor registers (TLReg0-TLReg31), each 1024 bytes in size. These registers are separate from the integer and floating-point register files.

#### 3.1.1. Register Naming Convention

- **Assembly syntax**: `t0` through `t31`
- **ABI names**: Same as assembly syntax
- **Encoding**: 5-bit field (0-31)

#### 3.1.2. Tensor Data Layout

Each TensorLoad register stores tensor data in row-major order. For a 4-dimensional tensor with shape [D0, D1, D2, D3], the linear address of element [i, j, k, l] is:

```
addr = i × (D1 × D2 × D3) + j × (D2 × D3) + k × D3 + l
```

### 3.2. Control and Status Registers

The TensorLoad extension defines several CSRs for configuration and control:

#### 3.2.1. Transpose Configuration CSRs

| CSR Name | Address | Description |
|----------|---------|-------------|
| `tl_tensor_shape` | 0x800 | Tensor shape configuration |
| `tl_transpose_perm` | 0x801 | Transpose permutation parameters |
| `tl_xpose_config` | 0x802 | Transpose configuration and control |
| `tl_xpose_status` | 0x803 | Transpose status and error information |

#### 3.2.2. Concatenation Configuration CSRs

| CSR Name | Address | Description |
|----------|---------|-------------|
| `tl_mask1_csr` | 0x810 | Valid bit mask 1 for concatenation |
| `tl_mask2_csr` | 0x811 | Valid bit mask 2 for concatenation |

#### 3.2.3. Memory Access Configuration CSRs

| CSR Name | Address | Description |
|----------|---------|-------------|
| `tl_load_mask_csr` | 0x812 | Load mask for conditional memory access |
| `tl_store_mask_csr` | 0x813 | Store mask for conditional memory access |

---

## 4. Instructions

### 4.1. Instruction Formats

The TensorLoad extension uses the CUSTOM-2 opcode space (0b1011011) and defines several instruction formats:

#### 4.1.1. R-type Format
```
31    25 24  20 19  15 14  12 11   7 6     0
+-------+-----+-----+-----+-----+---------+
| funct7| rs2 | rs1 |funct3| rd  | opcode  |
+-------+-----+-----+-----+-----+---------+
```

#### 4.1.2. I-type Format
```
31          20 19  15 14  12 11   7 6     0
+-------------+-----+-----+-----+---------+
|    imm      | rs1 |funct3| rd  | opcode  |
+-------------+-----+-----+-----+---------+
```

### 4.2. Tensor Transpose Instructions

#### 4.2.1. TL.XPOSE - 4D Tensor Transpose

**Format**: R-type  
**Syntax**: `tl.xpose rd, rs1, rs2`  
**Encoding**:

| 31-25 | 24-20 | 19-15 | 14-12 | 11-7 | 6-0 |
|-------|-------|-------|-------|------|-----|
| 0110000+dim_encoding | rs2 | rs1 | 000 | rd | 0b1011011 |

**Operation**: Transposes a 4-dimensional tensor by swapping two specified dimensions. The dimensions to swap are encoded in the funct7 field:

- `funct7[6:4] = 011` (TL.XPOSE identifier)
- `funct7[3:2] = dim1` (second dimension to swap)
- `funct7[1:0] = dim0` (first dimension to swap)

**Examples**:
- `tl.xpose.01 t1, t2, x10` - Swap dimensions 0 and 1
- `tl.xpose.23 t3, t4, x11` - Swap dimensions 2 and 3

### 4.3. Tensor Concatenation Instructions

#### 4.3.1. TL.CONCAT - Valid Bit Concatenation

**Format**: R-type  
**Syntax**: `tl.concat.dim rd, rs1, rs2`  
**Encoding**:

| 31-25 | 24-20 | 19-15 | 14-12 | 11-7 | 6-0 |
|-------|-------|-------|-------|------|-----|
| 1000000+concat_dim | rs2 | rs1 | 001 | rd | 0b1011011 |

**Operation**: Concatenates valid elements from two source tensors based on mask bits stored in CSRs.

**Algorithm**:
1. Read masks from `tl_mask1_csr` and `tl_mask2_csr`
2. Collect valid elements from rs1 where mask1[i] = 1
3. Collect valid elements from rs2 where mask2[i] = 1
4. Concatenate in order: {rs1_valid_data, rs2_valid_data}

### 4.4. Arithmetic Instructions

#### 4.4.1. TL.ADDI - Element-wise Immediate Addition

**Format**: I-type  
**Syntax**: `tl.addi rd, rs1, imm`  
**Encoding**:

| 31-20 | 19-15 | 14-12 | 11-7 | 6-0 |
|-------|-------|-------|------|-----|
| imm[11:0] | rs1 | 010 | rd | 0b1011011 |

**Operation**: Adds a 12-bit signed immediate value to each element in the source tensor with saturation.

**Saturation Logic**:
```c
result = element + sign_extend(imm)
if (result > 255) result = 255;
if (result < 0) result = 0;
```

### 4.5. Memory Access Instructions

#### 4.5.1. TL.MLOAD - Masked Load

**Format**: R-type  
**Syntax**: `tl.mload rd, rs1, rs2`  
**Encoding**:

| 31-25 | 24-20 | 19-15 | 14-12 | 11-7 | 6-0 |
|-------|-------|-------|-------|------|-----|
| 1010000 | rs2 | rs1 | 011 | rd | 0b1011011 |

**Operation**: Conditionally loads tensor slices from memory based on the highest dimension mask.

**Parameters**:
- `rs1`: GPR containing tensor dimension descriptor
- `rs2`: GPR containing memory base address
- `rd`: Target TLReg
- Mask: `tl_load_mask_csr`

#### 4.5.2. TL.MSTORE - Masked Store

**Format**: R-type  
**Syntax**: `tl.mstore rs1, rs2, rd`  
**Encoding**:

| 31-25 | 24-20 | 19-15 | 14-12 | 11-7 | 6-0 |
|-------|-------|-------|-------|------|-----|
| 1010001 | rs2 | rs1 | 100 | rd | 0b1011011 |

**Operation**: Conditionally stores tensor slices to memory based on the highest dimension mask.

**Parameters**:
- `rs1`: Source TLReg
- `rs2`: GPR containing memory base address
- `rd`: GPR containing tensor dimension descriptor
- Mask: `tl_store_mask_csr`

---

## 5. Standard TensorLoad Extensions

### 5.1. Ztlx: Tensor Transpose Extension

Provides hardware-accelerated tensor transpose operations for multi-dimensional arrays.

**Instructions**:
- `tl.xpose.01` - Swap dimensions 0 and 1
- `tl.xpose.02` - Swap dimensions 0 and 2
- `tl.xpose.03` - Swap dimensions 0 and 3
- `tl.xpose.12` - Swap dimensions 1 and 2
- `tl.xpose.13` - Swap dimensions 1 and 3
- `tl.xpose.23` - Swap dimensions 2 and 3

### 5.2. Ztlc: Tensor Concatenation Extension

Provides efficient tensor concatenation with validity masking.

**Instructions**:
- `tl.concat.0` - Concatenate along dimension 0
- `tl.concat.1` - Concatenate along dimension 1
- `tl.concat.2` - Concatenate along dimension 2

### 5.3. Ztla: Tensor Arithmetic Extension

Provides element-wise arithmetic operations on tensors.

**Instructions**:
- `tl.addi` - Element-wise immediate addition with saturation

### 5.4. Ztlm: Tensor Memory Extension

Provides conditional memory access operations for sparse tensor processing.

**Instructions**:
- `tl.mload` - Masked conditional load
- `tl.mstore` - Masked conditional store

---

## 6. TensorLoad Instruction Listing

### 6.1. Instruction Summary Table

| Instruction | Format | funct3 | funct7 | Description |
|-------------|--------|--------|--------|-------------|
| tl.xpose.01 | R-type | 000 | 0110001 | Transpose dimensions 0,1 |
| tl.xpose.02 | R-type | 000 | 0110010 | Transpose dimensions 0,2 |
| tl.xpose.03 | R-type | 000 | 0110011 | Transpose dimensions 0,3 |
| tl.xpose.12 | R-type | 000 | 0111001 | Transpose dimensions 1,2 |
| tl.xpose.13 | R-type | 000 | 0111010 | Transpose dimensions 1,3 |
| tl.xpose.23 | R-type | 000 | 0111011 | Transpose dimensions 2,3 |
| tl.concat.0 | R-type | 001 | 1000000 | Concatenate along dimension 0 |
| tl.concat.1 | R-type | 001 | 1000001 | Concatenate along dimension 1 |
| tl.concat.2 | R-type | 001 | 1000010 | Concatenate along dimension 2 |
| tl.addi | I-type | 010 | - | Element-wise immediate addition |
| tl.mload | R-type | 011 | 1010000 | Masked conditional load |
| tl.mstore | R-type | 100 | 1010001 | Masked conditional store |

### 6.2. CSR Summary Table

| CSR Name | Address | Width | Description |
|----------|---------|-------|-------------|
| tl_tensor_shape | 0x800 | 32 | Tensor shape configuration |
| tl_transpose_perm | 0x801 | 32 | Transpose permutation parameters |
| tl_xpose_config | 0x802 | 32 | Transpose configuration |
| tl_xpose_status | 0x803 | 32 | Transpose status |
| tl_mask1_csr | 0x810 | 32 | Concatenation mask 1 |
| tl_mask2_csr | 0x811 | 32 | Concatenation mask 2 |
| tl_load_mask_csr | 0x812 | 32 | Load operation mask |
| tl_store_mask_csr | 0x813 | 32 | Store operation mask |

---

## Appendix A: Usage Examples

### A.1. Matrix Transpose Example

```assembly
# Transpose a [8,16,8,2] tensor by swapping dimensions 0 and 1
li x10, 0x02081008          # Dimension descriptor: D3=2, D2=8, D1=16, D0=8
tl.xpose.01 t1, t2, x10     # Result shape: [16,8,8,2]
```

### A.2. Tensor Concatenation Example

```assembly
# Concatenate tensors along dimension 2 with validity masks
li x20, 0x0000000C          # mask1: 0011 (positions 2,3 valid)
li x21, 0x00000003          # mask2: 1100 (positions 0,1 valid)
csrrw x0, tl_mask1_csr, x20
csrrw x0, tl_mask2_csr, x21
tl.concat.2 t10, t11, t12   # Concatenate valid elements
```

### A.3. Conditional Memory Access Example

```assembly
# Load sparse tensor data with mask
li x10, 0x01081008          # Tensor shape: [8,16,8,1]
li x11, 0x1000              # Memory base address
li x12, 0x000000CC          # Load mask: 0b11001100
csrrw x0, tl_load_mask_csr, x12
tl.mload t1, x10, x11       # Load only slices 0,1,4,5
```

---

*This specification is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0).*

*Copyright © 2024 TensorLoad Extension Contributors*
