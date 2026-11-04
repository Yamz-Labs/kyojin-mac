; excerpt of the AIR of MLX 0.32.4 rope_single_float16 (mlx.metallib, macosx_14_0_arm64 wheel), disassembled with llvm-dis; MLX is MIT licensed
define internal fastcc void @_Z16rope_single_implIDhEvPU9MTLdeviceKT_PU9MTLdeviceS0_RU11MTLconstantKifRU11MTLconstantKfRU11MTLconstantKlDv2_jSB_(ptr addrspace(1) noundef %0, ptr addrspace(1) noundef %1, ptr addrspace(2) noundef align 4 dereferenceable(4) %2, float noundef %3, ptr addrspace(2) noundef align 4 dereferenceable(4) %4, ptr addrspace(2) noundef align 8 dereferenceable(8) %5, <2 x i32> noundef %6, <2 x i32> noundef %7) unnamed_addr #2 {
  %9 = load float, ptr addrspace(2) %4, align 4, !tbaa !33
  %10 = load i32, ptr addrspace(2) %2, align 4, !tbaa !44
  %11 = tail call float @air.convert.f.f32.s.i32(i32 %10) #5
  %12 = fmul float %9, %11
  %13 = fmul float %12, %3
  %14 = tail call float @air.fast_cos.f32(float %13) #5
  %15 = tail call float @air.fast_sin.f32(float %13) #5
  %16 = load i8, ptr addrspace(2) @_ZL11traditional, align 1, !tbaa !28, !range !32
  %17 = icmp eq i8 %16, 0
  %18 = extractelement <2 x i32> %6, i64 0
  br i1 %17, label %27, label %19

19:                                               ; preds = %8
  %20 = shl i32 %18, 1
  %21 = extractelement <2 x i32> %6, i64 1
  %22 = load i64, ptr addrspace(2) %5, align 8, !tbaa !46
  %23 = trunc i64 %22 to i32
  %24 = mul i32 %21, %23
  %25 = add i32 %24, %20
  %26 = add i32 %25, 1
  br label %35

27:                                               ; preds = %8
  %28 = extractelement <2 x i32> %6, i64 1
  %29 = load i64, ptr addrspace(2) %5, align 8, !tbaa !46
  %30 = trunc i64 %29 to i32
  %31 = mul i32 %28, %30
  %32 = add i32 %31, %18
  %33 = extractelement <2 x i32> %7, i64 0
  %34 = add i32 %32, %33
  br label %35

35:                                               ; preds = %27, %19
  %36 = phi i32 [ %26, %19 ], [ %34, %27 ]
  %37 = phi i32 [ %25, %19 ], [ %32, %27 ]
  %38 = zext i32 %37 to i64
  %39 = getelementptr inbounds half, ptr addrspace(1) %0, i64 %38
  %40 = load half, ptr addrspace(1) %39, align 2, !tbaa !48
  %41 = fpext half %40 to float
  %42 = zext i32 %36 to i64
  %43 = getelementptr inbounds half, ptr addrspace(1) %0, i64 %42
  %44 = load half, ptr addrspace(1) %43, align 2, !tbaa !48
  %45 = fpext half %44 to float
  %46 = load i8, ptr addrspace(2) @_ZL7forward, align 1, !tbaa !28, !range !32
  %47 = icmp eq i8 %46, 0
  br i1 %47, label %54, label %48

48:                                               ; preds = %35
  %49 = fneg float %15
  %50 = fmul float %49, %45
  %51 = tail call float @llvm.fmuladd.f32(float %41, float %14, float %50)
  %52 = fmul float %14, %45
  %53 = tail call float @llvm.fmuladd.f32(float %41, float %15, float %52)
  br label %60

54:                                               ; preds = %35
  %55 = fmul float %14, %41
  %56 = tail call float @llvm.fmuladd.f32(float %45, float %15, float %55)
  %57 = fneg float %15
  %58 = fmul float %57, %41
  %59 = tail call float @llvm.fmuladd.f32(float %45, float %14, float %58)
  br label %60

60:                                               ; preds = %54, %48
  %61 = phi float [ %51, %48 ], [ %56, %54 ]
  %62 = phi float [ %53, %48 ], [ %59, %54 ]
  %63 = fptrunc float %61 to half
  %64 = getelementptr inbounds half, ptr addrspace(1) %1, i64 %38
  store half %63, ptr addrspace(1) %64, align 2, !tbaa !48
  %65 = fptrunc float %62 to half
  %66 = getelementptr inbounds half, ptr addrspace(1) %1, i64 %42
  store half %65, ptr addrspace(1) %66, align 2, !tbaa !48
  ret void
}
