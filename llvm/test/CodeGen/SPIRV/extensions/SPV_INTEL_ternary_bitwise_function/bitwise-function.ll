; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s --spirv-ext=+SPV_INTEL_ternary_bitwise_function -o - | FileCheck %s --check-prefix=CHECK-EXTENSION
; RUN: not llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o %t.spvt 2>&1 | FileCheck %s --check-prefix=CHECK-NO-EXTENSION
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s --spirv-ext=+SPV_KHR_SPV_INTEL_ternary_bitwise_function -o - -filetype=obj | spirv-val %} 
;
; CHECK-NO-EXTENSION: LLVM ERROR: OpBitwiseFunctionINTEL instruction requires the following SPIR-V extension: SPV_INTEL_ternary_bitwise_function
;
; CHECK-EXTENSION-NOT: Name [[#]] "_Z28__spirv_BitwiseFunctionINTELiiij"
; CHECK-EXTENSION-NOT: Name [[#]] "_Z28__spirv_BitwiseFunctionINTELDv4_iS_S_j"
;
; CHECK-EXTENSION-DAG: Capability TernaryBitwiseFunctionINTEL
; CHECK-EXTENSION-DAG: Extension "SPV_INTEL_ternary_bitwise_function"
; CHECK-EXTENSION-DAG: TypeInt [[#TYPEINT:]] 32 0
; CHECK-EXTENSION-DAG: TypeVector [[#TYPEINTVEC4:]] [[#TYPEINT]] 4
; CHECK-EXTENSION-DAG: Constant [[#TYPEINT]] [[#ScalarLUT:]] 24
; CHECK-EXTENSION-DAG: Constant [[#TYPEINT]] [[#VecLUT:]] 42
; CHECK-EXTENSION: Load [[#TYPEINT]] [[#ScalarA:]]
; CHECK-EXTENSION: Load [[#TYPEINT]] [[#ScalarB:]]
; CHECK-EXTENSION: Load [[#TYPEINT]] [[#ScalarC:]]
; CHECK-EXTENSION: BitwiseFunctionINTEL [[#TYPEINT]] {{.*}} [[#ScalarA]] [[#ScalarB]] [[#ScalarC]] [[#ScalarLUT]]
; CHECK-EXTENSION: Load [[#TYPEINTVEC4]] [[#VecA:]]
; CHECK-EXTENSION: Load [[#TYPEINTVEC4]] [[#VecB:]]
; CHECK-EXTENSION: Load [[#TYPEINTVEC4]] [[#VecC:]]
; CHECK-EXTENSION: BitwiseFunctionINTEL [[#TYPEINTVEC4]] {{.*}} [[#VecA]] [[#VecB]] [[#VecC]] [[#VecLUT]]

; Function Attrs: nounwind readnone
define spir_kernel void @fooScalar() {
entry:
  %argA = alloca i32
  %argB = alloca i32
  %argC = alloca i32
  %A = load i32, ptr %argA
  %B = load i32, ptr %argB
  %C = load i32, ptr %argC
  %res = call spir_func i32 @_Z28__spirv_BitwiseFunctionINTELiiii(i32 %A, i32 %B, i32 %C, i32 24)
  ret void
}

; Function Attrs: nounwind readnone
define spir_kernel void @fooVec() {
entry:
  %argA = alloca <4 x i32>
  %argB = alloca <4 x i32>
  %argC = alloca <4 x i32>
  %A = load <4 x i32>, ptr %argA
  %B = load <4 x i32>, ptr %argB
  %C = load <4 x i32>, ptr %argC
  %res = call spir_func <4 x i32> @_Z28__spirv_BitwiseFunctionINTELDv4_iS_S_i(<4 x i32> %A, <4 x i32> %B, <4 x i32> %C, i32 42)
  ret void
}

declare dso_local spir_func i32 @_Z28__spirv_BitwiseFunctionINTELiiii(i32, i32, i32, i32)
declare dso_local spir_func <4 x i32> @_Z28__spirv_BitwiseFunctionINTELDv4_iS_S_i(<4 x i32>, <4 x i32>, <4 x i32>, i32)

!llvm.module.flags = !{!0}
!opencl.spir.version = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}