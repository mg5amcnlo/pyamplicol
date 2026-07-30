// SPDX-License-Identifier: 0BSD

//! Standard SymJIT P-kernel loading, binding, and execution.
//!
//! The generic raw-plane extension applied to SymJIT 2.22 exposes the existing
//! P-kernel machine code through a stable `#[repr(C)]` descriptor. Rusticol
//! keeps those descriptors in a cold-built vector, so duplicate inputs and
//! intentional input/output aliases never require overlapping Rust
//! references. Scalar/SIMD dispatch and descriptor rebinding allocate nothing.

use crate::{RusticolError, RusticolResult};
use std::any::Any;
use std::marker::PhantomData;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use symjit::{
    Applet, Application, Compiled, CompiledPlaneFunc, Compiler, CompilerType, Config, Defuns,
    PlaneDescriptor as SymjitPlaneDescriptor, Storage,
};

const SYMJIT_APPLICATION_MAGIC: usize = 0x4056_8795_410d_08e9;
const SYMJIT_APPLICATION_STORAGE_VERSION: usize = 3;
const MAX_OPTIMIZATION_LEVEL: u8 = 3;

type PlaneKernelFn = CompiledPlaneFunc<f64>;

/// Logical shape of a plane-oriented SymJIT application.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SymjitPlaneLayout {
    input_count: usize,
    output_count: usize,
    complex: bool,
}

impl SymjitPlaneLayout {
    #[cfg(test)]
    pub(crate) const fn real(input_count: usize, output_count: usize) -> Self {
        Self {
            input_count,
            output_count,
            complex: false,
        }
    }

    pub(crate) const fn complex(input_count: usize, output_count: usize) -> Self {
        Self {
            input_count,
            output_count,
            complex: true,
        }
    }

    pub(crate) const fn is_complex(self) -> bool {
        self.complex
    }

    pub(crate) fn input_plane_count(self) -> RusticolResult<usize> {
        self.input_count
            .checked_mul(if self.complex { 2 } else { 1 })
            .ok_or_else(|| {
                RusticolError::invalid_argument("SymJIT input plane count overflows usize")
            })
    }

    pub(crate) fn output_plane_count(self) -> RusticolResult<usize> {
        self.output_count
            .checked_mul(if self.complex { 2 } else { 1 })
            .ok_or_else(|| {
                RusticolError::invalid_argument("SymJIT output plane count overflows usize")
            })
    }

    pub(crate) fn plane_count(self) -> RusticolResult<usize> {
        let components = if self.complex { 2 } else { 1 };
        self.input_count
            .checked_add(self.output_count)
            .and_then(|count| count.checked_mul(components))
            .ok_or_else(|| RusticolError::invalid_argument("SymJIT plane count overflows usize"))
    }
}

/// A lifetime-tracked wrapper around SymJIT's stable raw plane descriptor.
#[derive(Debug)]
#[repr(transparent)]
pub(crate) struct SymjitRawPlane<'a> {
    descriptor: SymjitPlaneDescriptor<f64>,
    _lifetime: PhantomData<&'a mut [f64]>,
}

/// Single cold-path descriptor type used by Rusticol schedulers.
pub(crate) type PlaneDescriptor<'a> = SymjitRawPlane<'a>;

impl<'a> SymjitRawPlane<'a> {
    #[cfg(test)]
    pub(crate) fn from_slice(values: &'a mut [f64]) -> Self {
        Self {
            // SAFETY: The returned wrapper retains the exclusive slice
            // lifetime, so the allocation outlives every use of the raw
            // descriptor.
            descriptor: unsafe {
                SymjitPlaneDescriptor::from_raw_parts(values.as_mut_ptr(), values.len())
            },
            _lifetime: PhantomData,
        }
    }

    /// Creates a cold-binding descriptor without creating a Rust reference.
    ///
    /// # Safety
    ///
    /// For the returned lifetime, `pointer` must remain allocated, aligned,
    /// and writable for `len` consecutive `f64` values. The memory must not be
    /// accessed concurrently with an eventual bound kernel. Duplicate
    /// descriptors and intentional input/output aliases are permitted because
    /// this wrapper never creates overlapping Rust references.
    pub(crate) unsafe fn from_raw_parts(pointer: *mut f64, len: usize) -> RusticolResult<Self> {
        let pointer = if len == 0 && pointer.is_null() {
            NonNull::<f64>::dangling().as_ptr()
        } else {
            NonNull::new(pointer)
                .ok_or_else(|| {
                    RusticolError::invalid_argument(
                        "a non-empty SymJIT plane has a null data pointer",
                    )
                })?
                .as_ptr()
        };
        if pointer.addr() % std::mem::align_of::<f64>() != 0 {
            return Err(RusticolError::invalid_argument(
                "a SymJIT plane data pointer is not aligned for f64",
            ));
        }
        Ok(Self {
            descriptor: unsafe { SymjitPlaneDescriptor::from_raw_parts(pointer, len) },
            _lifetime: PhantomData,
        })
    }

    fn interval(&self) -> RusticolResult<Option<(usize, usize)>> {
        if self.descriptor.len == 0 {
            return Ok(None);
        }
        let start = self.descriptor.data.addr();
        let bytes = self
            .descriptor
            .len
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                RusticolError::invalid_argument("SymJIT plane byte length overflows usize")
            })?;
        let end = start.checked_add(bytes).ok_or_else(|| {
            RusticolError::invalid_argument("SymJIT plane address range overflows usize")
        })?;
        Ok(Some((start, end)))
    }

    pub(crate) const fn len(&self) -> usize {
        self.descriptor.len
    }
}

/// Loaded P-kernel code. The `Applet` owns the executable mappings referenced
/// by the cached function pointers.
pub(crate) struct SymjitPlaneKernel {
    applet: Applet,
    display_path: PathBuf,
    layout: SymjitPlaneLayout,
    scalar: PlaneKernelFn,
    simd: Option<PlaneKernelFn>,
    simd_lanes: usize,
}

impl std::fmt::Debug for SymjitPlaneKernel {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SymjitPlaneKernel")
            .field("display_path", &self.display_path)
            .field("layout", &self.layout)
            .field("simd_lanes", &self.simd_lanes)
            .finish_non_exhaustive()
    }
}

impl SymjitPlaneKernel {
    /// Loads, authenticates, recompiles, and seals a storage-v3 P-kernel.
    pub(crate) fn load_bytes(
        bytes: &[u8],
        display_path: impl Into<PathBuf>,
        layout: SymjitPlaneLayout,
    ) -> RusticolResult<Self> {
        let display_path = display_path.into();
        authenticate_storage_v3(bytes, &display_path)?;

        let mut loader_config = explicit_loader_config()?;
        loader_config.set_defuns(Defuns::new());
        let mut input = bytes;
        let mut application = guard_symjit_panic(
            || Application::load(&mut input, &loader_config),
            || format!("load plane application {}", display_path.display()),
        )?
        .map_err(|error| {
            RusticolError::compatibility(format!(
                "could not load SymJIT storage-v3 plane application {}: {error}",
                display_path.display()
            ))
        })?;
        if !input.is_empty() {
            return Err(RusticolError::integrity(format!(
                "SymJIT plane application {} has {} unauthenticated trailing bytes",
                display_path.display(),
                input.len()
            )));
        }
        validate_application(&application, &display_path, layout)?;

        // SIMD compilation is lazy in SymJIT. Calling this before sealing
        // guarantees that a supported SIMD kernel is retained by the Applet.
        guard_symjit_panic(
            || application.prepare_simd(),
            || {
                format!(
                    "prepare SIMD for plane application {}",
                    display_path.display()
                )
            },
        )?;
        validate_compiled_kernels(&application, &display_path)?;

        let applet = guard_symjit_panic(
            || application.seal(),
            || format!("seal plane application {}", display_path.display()),
        )?
        .map_err(|error| {
            RusticolError::evaluation(format!(
                "could not seal SymJIT plane application {}: {error}",
                display_path.display()
            ))
        })?;
        let scalar = applet.scalar_plane_kernel().ok_or_else(|| {
            RusticolError::compatibility(format!(
                "SymJIT plane application {} has no scalar raw-plane kernel",
                display_path.display()
            ))
        })?;
        let (simd, simd_lanes) = match applet.compiled_simd.as_ref() {
            Some(compiled) if compiled.support_indirect() => {
                (applet.simd_plane_kernel(), compiled.count_lanes())
            }
            _ => (None, 1),
        };
        validate_simd_lane_width(simd.is_some(), simd_lanes, &display_path)?;

        Ok(Self {
            applet,
            display_path,
            layout,
            scalar,
            simd,
            simd_lanes,
        })
    }

    pub(crate) fn input_plane_count(&self) -> usize {
        self.layout
            .input_plane_count()
            .expect("validated SymJIT input plane count")
    }

    pub(crate) fn output_plane_count(&self) -> usize {
        self.layout
            .output_plane_count()
            .expect("validated SymJIT output plane count")
    }

    #[cfg(test)]
    pub(crate) const fn simd_lanes(&self) -> usize {
        self.simd_lanes
    }

    pub(crate) fn optimization_level(&self) -> u8 {
        self.applet.config.opt_level()
    }

    /// Cold-binds safe, mutually disjoint slices into one stable descriptor
    /// table. The vector is never resized on the execution path.
    #[cfg(test)]
    pub(crate) fn bind<'kernel, 'planes>(
        &'kernel self,
        planes: Vec<&'planes mut [f64]>,
    ) -> RusticolResult<SymjitBoundPlaneKernel<'kernel, 'planes>> {
        let table = self.build_table(planes)?;
        Ok(SymjitBoundPlaneKernel {
            kernel: self,
            table,
        })
    }

    /// Cold-builds a descriptor table which is independent of the kernel
    /// borrow and can be cached by recurrence/eager runtimes.
    #[cfg(test)]
    pub(crate) fn build_table<'planes>(
        &self,
        planes: Vec<&'planes mut [f64]>,
    ) -> RusticolResult<SymjitPlaneTable<'planes>> {
        validate_slice_table(self.layout, &planes)?;
        let point_count = planes.first().map_or(0, |plane| plane.len());
        let descriptors = planes
            .into_iter()
            .map(|plane| {
                // SAFETY: The table carries `'planes`, retaining each
                // exclusive slice borrow for the descriptor's lifetime.
                unsafe { SymjitPlaneDescriptor::from_raw_parts(plane.as_mut_ptr(), plane.len()) }
            })
            .collect();
        Ok(SymjitPlaneTable {
            descriptors,
            point_count,
            input_plane_count: self.input_plane_count(),
            output_plane_count: self.output_plane_count(),
            _lifetime: PhantomData,
        })
    }

    /// Cold-binds raw planes after validating their shape and address ranges.
    /// Duplicate input planes and intentional input/output aliases are valid.
    #[cfg(test)]
    pub(crate) fn bind_raw<'kernel, 'planes>(
        &'kernel self,
        planes: Vec<SymjitRawPlane<'planes>>,
    ) -> RusticolResult<SymjitBoundPlaneKernel<'kernel, 'planes>> {
        let table = self.build_raw_table(planes)?;
        Ok(SymjitBoundPlaneKernel {
            kernel: self,
            table,
        })
    }

    /// Cold-builds a reusable table from raw plane descriptors.
    pub(crate) fn build_raw_table<'planes>(
        &self,
        planes: Vec<PlaneDescriptor<'planes>>,
    ) -> RusticolResult<SymjitPlaneTable<'planes>> {
        self.build_raw_table_from_descriptors(&planes)
    }

    /// Cold-builds a reusable table while retaining the caller's descriptor
    /// vector and capacity for subsequent allocation-free row preparation.
    pub(crate) fn build_raw_table_from_descriptors<'planes>(
        &self,
        planes: &[PlaneDescriptor<'planes>],
    ) -> RusticolResult<SymjitPlaneTable<'planes>> {
        validate_raw_table(self.layout, planes)?;
        let point_count = planes.first().map_or(0, PlaneDescriptor::len);
        Ok(SymjitPlaneTable {
            descriptors: planes.iter().map(|plane| plane.descriptor).collect(),
            point_count,
            input_plane_count: self.input_plane_count(),
            output_plane_count: self.output_plane_count(),
            _lifetime: PhantomData,
        })
    }

    pub(crate) fn execute_table(
        &self,
        table: &mut SymjitPlaneTable<'_>,
        point_start: usize,
        point_count: usize,
    ) -> RusticolResult<()> {
        if table.input_plane_count != self.input_plane_count()
            || table.output_plane_count != self.output_plane_count()
        {
            return Err(RusticolError::invalid_argument(format!(
                "SymJIT plane table shape ({}, {}) does not match kernel shape ({}, {})",
                table.input_plane_count,
                table.output_plane_count,
                self.input_plane_count(),
                self.output_plane_count()
            )));
        }
        self.invoke_descriptors(table, point_start, point_count)
    }

    /// Validates a scheduler-owned raw descriptor slice on the cold path.
    ///
    /// Retained as the checked counterpart required by the generic unsafe raw
    /// callable contract. Current schedulers perform equivalent validation
    /// while constructing their authenticated row storage.
    #[allow(dead_code)]
    pub(crate) fn validate_raw_descriptors(
        &self,
        planes: &[PlaneDescriptor<'_>],
    ) -> RusticolResult<usize> {
        validate_raw_table(self.layout, planes)?;
        Ok(planes.first().map_or(0, PlaneDescriptor::len))
    }

    /// Executes a scheduler-owned descriptor slice without allocating or
    /// copying its entries.
    ///
    /// # Safety
    ///
    /// [`Self::validate_raw_descriptors`] must previously have succeeded for
    /// the same descriptor shape. Every pointed-to allocation must still be
    /// live, synchronized, and cover `point_start..point_start + point_count`.
    pub(crate) unsafe fn execute_raw_descriptors_unchecked(
        &self,
        planes: &[PlaneDescriptor<'_>],
        point_start: usize,
        point_count: usize,
    ) -> RusticolResult<()> {
        if planes.len() != self.layout.plane_count()? {
            return Err(RusticolError::invalid_argument(
                "SymJIT raw descriptor shape changed after cold validation",
            ));
        }
        let point_end = point_start.checked_add(point_count).ok_or_else(|| {
            RusticolError::invalid_argument("SymJIT plane execution range overflows usize")
        })?;
        if point_end > planes.first().map_or(0, PlaneDescriptor::len) {
            return Err(RusticolError::invalid_argument(
                "SymJIT raw descriptor range exceeds its validated plane capacity",
            ));
        }
        if point_count == 0 {
            return Ok(());
        }
        // `SymjitRawPlane` is transparent over the upstream `#[repr(C)]`
        // descriptor; the lifetime marker has no storage.
        let table = planes.as_ptr().cast::<SymjitPlaneDescriptor<f64>>();
        let failure = invoke_range(
            self.scalar,
            self.simd,
            self.simd_lanes,
            table,
            self.applet.params.as_ptr(),
            point_start,
            point_end,
        )?;
        if let Some(failure) = failure {
            return Err(RusticolError::evaluation(format!(
                "SymJIT plane application {} returned status {} from its {} kernel at {} index {}",
                self.display_path.display(),
                failure.status,
                failure.kind.as_str(),
                failure.kind.index_name(),
                failure.index
            )));
        }
        Ok(())
    }

    /// Invokes a prebuilt, validated descriptor table without allocating or
    /// changing its address. This is the only runtime method coupled to
    /// SymJIT's raw P-kernel callable.
    fn invoke_descriptors(
        &self,
        table: &SymjitPlaneTable<'_>,
        point_start: usize,
        point_count: usize,
    ) -> RusticolResult<()> {
        let point_end = point_start.checked_add(point_count).ok_or_else(|| {
            RusticolError::invalid_argument("SymJIT plane execution range overflows usize")
        })?;
        if point_end > table.point_count {
            return Err(RusticolError::invalid_argument(format!(
                "SymJIT plane execution range {point_start}..{point_end} exceeds point capacity {}",
                table.point_count
            )));
        }
        if point_count == 0 {
            return Ok(());
        }

        let failure = invoke_range(
            self.scalar,
            self.simd,
            self.simd_lanes,
            table.as_symjit_ptr(),
            self.applet.params.as_ptr(),
            point_start,
            point_end,
        )?;
        if let Some(failure) = failure {
            return Err(RusticolError::evaluation(format!(
                "SymJIT plane application {} returned status {} from its {} kernel at {} index {}",
                self.display_path.display(),
                failure.status,
                failure.kind.as_str(),
                failure.kind.index_name(),
                failure.index
            )));
        }
        Ok(())
    }
}

/// The single representation-sensitive descriptor table. Descriptors are
/// allocated and validated on the cold path and never resized while bound.
pub(crate) struct SymjitPlaneTable<'planes> {
    descriptors: Vec<SymjitPlaneDescriptor<f64>>,
    point_count: usize,
    input_plane_count: usize,
    output_plane_count: usize,
    _lifetime: PhantomData<&'planes mut [f64]>,
}

impl SymjitPlaneTable<'_> {
    fn as_symjit_ptr(&self) -> *const SymjitPlaneDescriptor<f64> {
        self.descriptors.as_ptr()
    }

    #[cfg(test)]
    pub(crate) const fn point_count(&self) -> usize {
        self.point_count
    }
}

#[cfg(test)]
impl<'planes> SymjitPlaneTable<'planes> {
    /// Rebinds the existing descriptor allocation to a new set of raw planes.
    ///
    /// The vector allocation and address remain unchanged. Validation finishes
    /// before any slot is replaced.
    ///
    /// # Safety
    ///
    /// Every descriptor must remain allocated and writable for `'planes`. No
    /// execution may overlap this call, the old descriptors must no longer be
    /// used after rebinding begins, and the caller must synchronize access to
    /// the new storage until this table is rebound again or dropped. Aliases
    /// must be valid for the particular P-kernel operation being invoked.
    pub(crate) unsafe fn rebind_raw(
        &mut self,
        planes: &[PlaneDescriptor<'planes>],
    ) -> RusticolResult<()> {
        validate_raw_plane_count_and_ranges(self.descriptors.len(), planes)?;

        for (descriptor, plane) in self.descriptors.iter_mut().zip(planes) {
            *descriptor = plane.descriptor;
        }
        self.point_count = planes.first().map_or(0, PlaneDescriptor::len);
        Ok(())
    }
}

/// A stable, allocation-free hot binding of a P-kernel to split planes.
#[cfg(test)]
pub(crate) struct SymjitBoundPlaneKernel<'kernel, 'planes> {
    // Keep the Applet reachable for at least as long as every machine-code
    // function pointer invocation.
    kernel: &'kernel SymjitPlaneKernel,
    table: SymjitPlaneTable<'planes>,
}

#[cfg(test)]
impl std::fmt::Debug for SymjitBoundPlaneKernel<'_, '_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SymjitBoundPlaneKernel")
            .field("kernel", &self.kernel)
            .field("plane_count", &self.table.descriptors.len())
            .field("point_count", &self.table.point_count)
            .finish()
    }
}

#[cfg(test)]
impl SymjitBoundPlaneKernel<'_, '_> {
    pub(crate) const fn point_count(&self) -> usize {
        self.table.point_count
    }

    pub(crate) fn execute_all(&mut self) -> RusticolResult<()> {
        self.execute_range(0, self.table.point_count)
    }

    /// Executes `[point_start, point_start + point_count)`.
    ///
    /// Scalar calls receive a point index. SIMD calls receive a SIMD-block
    /// index, so unaligned heads and incomplete tails are dispatched through
    /// the scalar kernel.
    pub(crate) fn execute_range(
        &mut self,
        point_start: usize,
        point_count: usize,
    ) -> RusticolResult<()> {
        self.kernel
            .execute_table(&mut self.table, point_start, point_count)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CallKind {
    Scalar,
    Simd,
}

impl CallKind {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Simd => "SIMD",
        }
    }

    const fn index_name(self) -> &'static str {
        match self {
            Self::Scalar => "point",
            Self::Simd => "block",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct StatusFailure {
    kind: CallKind,
    index: usize,
    status: i32,
}

fn invoke_range(
    scalar: PlaneKernelFn,
    simd: Option<PlaneKernelFn>,
    simd_lanes: usize,
    table: *const SymjitPlaneDescriptor<f64>,
    params: *const f64,
    point_start: usize,
    point_end: usize,
) -> RusticolResult<Option<StatusFailure>> {
    catch_unwind(AssertUnwindSafe(|| {
        let mut point = point_start;

        if let Some(simd) = simd.filter(|_| simd_lanes > 1) {
            while point < point_end && !point.is_multiple_of(simd_lanes) {
                // SAFETY: The caller has validated every descriptor and the
                // point range; all allocations remain live for this call.
                let status = unsafe { scalar(std::ptr::null(), table, point, params) };
                if status != 0 {
                    return Some(StatusFailure {
                        kind: CallKind::Scalar,
                        index: point,
                        status,
                    });
                }
                point += 1;
            }

            while point_end - point >= simd_lanes {
                let block = point / simd_lanes;
                // SAFETY: As above; SIMD uses a block index covering only the
                // validated aligned interior.
                let status = unsafe { simd(std::ptr::null(), table, block, params) };
                if status != 0 {
                    return Some(StatusFailure {
                        kind: CallKind::Simd,
                        index: block,
                        status,
                    });
                }
                point += simd_lanes;
            }
        }

        while point < point_end {
            // SAFETY: The scalar tail lies inside the validated point range.
            let status = unsafe { scalar(std::ptr::null(), table, point, params) };
            if status != 0 {
                return Some(StatusFailure {
                    kind: CallKind::Scalar,
                    index: point,
                    status,
                });
            }
            point += 1;
        }
        None
    }))
    .map_err(|payload| {
        RusticolError::evaluation(format!(
            "SymJIT panicked while executing a plane kernel: {}",
            panic_detail(payload)
        ))
    })
}

/// Compiles `repr(Evaluator.get_instructions())` into a complex, SIMD-enabled,
/// direct-arena SymJIT storage-v3 application.
///
/// This is deliberately a cold-path function. It constructs every SymJIT
/// option explicitly so a process-local `symjit.toml` cannot alter generated
/// artifacts.
pub fn compile_symbolica_program_to_plane_application_bytes(
    program_repr: &str,
    input_complex_count: usize,
    output_complex_count: usize,
    opt_level: u8,
    compress: bool,
) -> RusticolResult<Vec<u8>> {
    if program_repr.trim().is_empty() {
        return Err(RusticolError::invalid_argument(
            "Symbolica instruction representation must not be empty",
        ));
    }
    let layout = SymjitPlaneLayout::complex(input_complex_count, output_complex_count);
    let config = explicit_plane_config(true, opt_level, compress)?;
    let mut compiler = Compiler::with_config(config);
    let mut application = guard_symjit_panic(
        || compiler.translate(program_repr.to_owned(), input_complex_count),
        || "translate Symbolica instructions to a SymJIT plane application".to_string(),
    )?
    .map_err(|error| {
        RusticolError::serialization(format!(
            "could not translate Symbolica instructions to a SymJIT plane application: {error}"
        ))
    })?;
    let display_path = Path::new("<generated Symbolica plane application>");
    validate_application(&application, display_path, layout)?;
    guard_symjit_panic(
        || application.prepare_simd(),
        || "prepare SIMD for generated SymJIT plane application".to_string(),
    )?;
    validate_compiled_kernels(&application, display_path)?;

    let mut bytes = Vec::new();
    guard_symjit_panic(
        || application.save(&mut bytes),
        || "serialize generated SymJIT plane application".to_string(),
    )?
    .map_err(|error| {
        RusticolError::serialization(format!(
            "could not serialize generated SymJIT plane application: {error}"
        ))
    })?;
    authenticate_storage_v3(&bytes, display_path)?;
    Ok(bytes)
}

fn explicit_plane_config(complex: bool, opt_level: u8, compress: bool) -> RusticolResult<Config> {
    if opt_level > MAX_OPTIMIZATION_LEVEL {
        return Err(RusticolError::invalid_argument(format!(
            "SymJIT optimization level {opt_level} is unsupported; expected 0 through {MAX_OPTIMIZATION_LEVEL}"
        )));
    }
    let mut config = Config::new(CompilerType::Native, 0).map_err(|error| {
        RusticolError::internal(format!(
            "could not construct explicit SymJIT native configuration: {error}"
        ))
    })?;
    config.set_opt_level(opt_level);
    config.set_cse(true);
    config.set_fastmath(true);
    config.set_simd(true);
    // pyAmpliCol's portable direct-plane contract is f64x4 on x86 and
    // f64x2 on AArch64. AVX-512 remains an explicit future ABI.
    config.enable_simd512(false);
    config.set_simd_branch(false);
    config.set_complex(complex);
    config.set_fast_complex(false);
    config.set_threads(false);
    config.set_symbolica(true);
    config.set_compact(true);
    config.set_compress(compress);
    config.set_direct(false);
    config.set_huge(false);
    config.set_parallel_mul(true);
    config.set_direct_arena(true);
    // Standard P-kernels own only an identity overwrite. pyAmpliCol applies
    // factors and accumulation in its alias-safe arena epilogues. Set these
    // currently latent 2.22 options explicitly so a future SymJIT
    // implementation cannot silently change persisted application semantics.
    config.set_direct_arena_operation(0);
    config.set_direct_arena_identity_output(true);
    Ok(config)
}

fn explicit_loader_config() -> RusticolResult<Config> {
    let mut config = Config::new(CompilerType::Native, 0).map_err(|error| {
        RusticolError::internal(format!(
            "could not construct explicit SymJIT loader configuration: {error}"
        ))
    })?;
    config.set_opt_level(0);
    config.set_cse(false);
    config.set_fastmath(false);
    config.set_simd(false);
    config.enable_simd512(false);
    config.set_simd_branch(false);
    config.set_complex(false);
    config.set_fast_complex(false);
    config.set_threads(false);
    config.set_symbolica(false);
    config.set_compact(false);
    config.set_compress(false);
    config.set_direct(false);
    config.set_huge(false);
    config.set_parallel_mul(false);
    config.set_direct_arena(false);
    Ok(config)
}

fn authenticate_storage_v3(bytes: &[u8], display_path: &Path) -> RusticolResult<()> {
    let word = std::mem::size_of::<usize>();
    let header_len = word
        .checked_mul(2)
        .ok_or_else(|| RusticolError::internal("SymJIT storage header length overflows usize"))?;
    if word != 8 {
        return Err(RusticolError::compatibility(
            "SymJIT storage-v3 plane applications require a 64-bit target",
        ));
    }
    if bytes.len() < header_len {
        return Err(RusticolError::integrity(format!(
            "SymJIT plane application {} is truncated before its storage-v3 header",
            display_path.display()
        )));
    }
    let magic = usize::from_le_bytes(
        bytes[..word]
            .try_into()
            .map_err(|_| RusticolError::internal("could not decode SymJIT application magic"))?,
    );
    if magic != SYMJIT_APPLICATION_MAGIC {
        return Err(RusticolError::compatibility(format!(
            "SymJIT plane application {} has invalid application magic",
            display_path.display()
        )));
    }
    let version = usize::from_le_bytes(
        bytes[word..header_len]
            .try_into()
            .map_err(|_| RusticolError::internal("could not decode SymJIT storage version"))?,
    );
    if version != SYMJIT_APPLICATION_STORAGE_VERSION {
        return Err(RusticolError::compatibility(format!(
            "SymJIT plane application {} uses storage version {version}, expected {SYMJIT_APPLICATION_STORAGE_VERSION}",
            display_path.display()
        )));
    }
    Ok(())
}

fn validate_application(
    application: &Application,
    display_path: &Path,
    layout: SymjitPlaneLayout,
) -> RusticolResult<()> {
    let config = &application.config;
    if config.is_enabled_simd512() {
        return Err(RusticolError::compatibility(format!(
            "SymJIT plane application {} enables unsupported SIMD512 code generation; \
             regenerate it for the portable scalar/f64x2/f64x4 arena ABI",
            display_path.display()
        )));
    }
    if !config.symbolica()
        || !config.direct_arena()
        || config.direct()
        || config.direct_arena_operation() != 0
        || !config.direct_arena_identity_output()
    {
        return Err(RusticolError::compatibility(format!(
            "SymJIT application {} is not an identity-overwrite indirect Symbolica \
             direct-arena P-kernel",
            display_path.display()
        )));
    }
    if config.is_complex() != layout.is_complex() {
        return Err(RusticolError::integrity(format!(
            "SymJIT plane application {} complex mode does not match its binding layout",
            display_path.display()
        )));
    }
    let input_plane_count = layout.input_plane_count()?;
    let output_plane_count = layout.output_plane_count()?;
    if application.count_states != input_plane_count {
        return Err(RusticolError::integrity(format!(
            "SymJIT plane application {} stores {} input planes, expected {}",
            display_path.display(),
            application.count_states,
            input_plane_count
        )));
    }
    if application.count_obs != output_plane_count {
        return Err(RusticolError::integrity(format!(
            "SymJIT plane application {} stores {} output planes, expected {}",
            display_path.display(),
            application.count_obs,
            output_plane_count
        )));
    }
    if application.count_params != 0 {
        return Err(RusticolError::compatibility(format!(
            "SymJIT plane application {} retains {} scalar parameters; direct-arena inputs must all be states",
            display_path.display(),
            application.count_params
        )));
    }
    if application.count_diffs != 0 {
        return Err(RusticolError::compatibility(format!(
            "SymJIT plane application {} contains differential outputs",
            display_path.display()
        )));
    }
    Ok(())
}

fn validate_simd_lane_width(
    has_simd_kernel: bool,
    simd_lanes: usize,
    display_path: &Path,
) -> RusticolResult<()> {
    let valid = if has_simd_kernel {
        matches!(simd_lanes, 2 | 4)
    } else {
        simd_lanes == 1
    };
    if !valid {
        return Err(RusticolError::compatibility(format!(
            "SymJIT plane application {} reports unsupported SIMD lane width {simd_lanes}; \
             expected scalar fallback, f64x2, or f64x4",
            display_path.display()
        )));
    }
    Ok(())
}

fn validate_compiled_kernels(application: &Application, display_path: &Path) -> RusticolResult<()> {
    let scalar = application.compiled.as_ref().ok_or_else(|| {
        RusticolError::compatibility(format!(
            "SymJIT plane application {} has no scalar native kernel",
            display_path.display()
        ))
    })?;
    if !scalar.support_indirect() {
        return Err(RusticolError::compatibility(format!(
            "SymJIT plane application {} scalar kernel does not support indirect plane descriptors",
            display_path.display()
        )));
    }
    if let Some(simd) = application.compiled_simd.as_ref()
        && (!simd.support_indirect() || simd.count_lanes() <= 1)
    {
        return Err(RusticolError::compatibility(format!(
            "SymJIT plane application {} has an invalid indirect SIMD kernel",
            display_path.display()
        )));
    }
    Ok(())
}

#[cfg(test)]
fn validate_slice_table(layout: SymjitPlaneLayout, planes: &[&mut [f64]]) -> RusticolResult<()> {
    let expected = layout.plane_count()?;
    if planes.len() != expected {
        return Err(RusticolError::invalid_argument(format!(
            "SymJIT plane table has {} planes, expected {expected}",
            planes.len()
        )));
    }
    let point_count = planes.first().map_or(0, |plane| plane.len());
    if let Some((index, plane)) = planes
        .iter()
        .enumerate()
        .find(|(_, plane)| plane.len() != point_count)
    {
        return Err(RusticolError::invalid_argument(format!(
            "SymJIT plane {index} has length {}, expected {point_count}",
            plane.len()
        )));
    }
    Ok(())
}

fn validate_raw_table(
    layout: SymjitPlaneLayout,
    planes: &[SymjitRawPlane<'_>],
) -> RusticolResult<()> {
    let expected = layout.plane_count()?;
    validate_raw_plane_count_and_ranges(expected, planes)
}

fn validate_raw_plane_count_and_ranges(
    expected: usize,
    planes: &[SymjitRawPlane<'_>],
) -> RusticolResult<()> {
    if planes.len() != expected {
        return Err(RusticolError::invalid_argument(format!(
            "SymJIT raw plane table has {} planes, expected {expected}",
            planes.len()
        )));
    }
    let point_count = planes.first().map_or(0, PlaneDescriptor::len);
    for (index, plane) in planes.iter().enumerate() {
        if plane.len() != point_count {
            return Err(RusticolError::invalid_argument(format!(
                "SymJIT raw plane {index} has length {}, expected {point_count}",
                plane.len()
            )));
        }
        let _ = plane.interval()?;
    }
    Ok(())
}

fn guard_symjit_panic<T>(
    operation: impl FnOnce() -> T,
    action: impl FnOnce() -> String,
) -> RusticolResult<T> {
    catch_unwind(AssertUnwindSafe(operation)).map_err(|payload| {
        RusticolError::compatibility(format!(
            "SymJIT panicked while trying to {}: {}",
            action(),
            panic_detail(payload)
        ))
    })
}

fn panic_detail(payload: Box<dyn Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "non-string panic payload".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RusticolErrorKind;

    const MODEL: &str = "
([('fun', ('temp', 0), 'square', [], [('param', 0)], False),
  ('add', ('temp', 1), [('temp', 0), ('param', 1)], 0),
  ('assign', ('out', 0), ('temp', 1))],
 2,
 [])
";
    const LENGTHS: &[usize] = &[1, 2, 3, 7, 8, 127, 128, 129, 1023, 1024, 1025];
    const SENTINEL: f64 = -9.876_543_210_123_456e211;

    fn application_bytes(complex: bool, opt_level: u8) -> Vec<u8> {
        let config = explicit_plane_config(complex, opt_level, true).unwrap();
        application_bytes_with_config(config)
    }

    fn application_bytes_with_config(config: Config) -> Vec<u8> {
        let mut compiler = Compiler::with_config(config);
        let mut application = compiler.translate(MODEL.to_string(), 2).unwrap();
        application.prepare_simd();
        let mut bytes = Vec::new();
        application.save(&mut bytes).unwrap();
        bytes
    }

    fn guarded_plane(len: usize, values: impl Fn(usize) -> f64) -> Vec<f64> {
        let mut plane = vec![SENTINEL; len + 2];
        for index in 0..len {
            plane[index + 1] = values(index);
        }
        plane
    }

    fn assert_guards(plane: &[f64]) {
        assert_eq!(plane.first().copied(), Some(SENTINEL));
        assert_eq!(plane.last().copied(), Some(SENTINEL));
    }

    #[test]
    fn real_scalar_and_simd_ranges_preserve_sentinels_for_all_lengths() {
        let bytes = application_bytes(false, 3);
        let kernel = SymjitPlaneKernel::load_bytes(
            &bytes,
            "real-contract.symjit",
            SymjitPlaneLayout::real(2, 1),
        )
        .unwrap();
        assert_eq!(kernel.input_plane_count(), 2);
        assert_eq!(kernel.output_plane_count(), 1);
        assert!(kernel.simd_lanes() >= 1);

        for &len in LENGTHS {
            let mut x = guarded_plane(len, |i| i as f64 * 0.25 - 3.0);
            let mut y = guarded_plane(len, |i| 7.0 - i as f64 * 0.125);
            let mut out = guarded_plane(len, |_| f64::NAN);
            {
                let mut bound = kernel
                    .bind(vec![
                        &mut x[1..len + 1],
                        &mut y[1..len + 1],
                        &mut out[1..len + 1],
                    ])
                    .unwrap();
                assert_eq!(bound.point_count(), len);
                bound.execute_all().unwrap();
            }
            for index in 0..len {
                let expected = x[index + 1] * x[index + 1] + y[index + 1];
                assert_eq!(out[index + 1], expected, "length {len}, point {index}");
            }
            assert_guards(&x);
            assert_guards(&y);
            assert_guards(&out);
        }
    }

    #[test]
    fn complex_split_planes_cover_scalar_simd_and_odd_tails() {
        let bytes =
            compile_symbolica_program_to_plane_application_bytes(MODEL, 2, 1, 3, true).unwrap();
        let kernel = SymjitPlaneKernel::load_bytes(
            &bytes,
            "complex-contract.symjit",
            SymjitPlaneLayout::complex(2, 1),
        )
        .unwrap();
        assert_eq!(kernel.input_plane_count(), 4);
        assert_eq!(kernel.output_plane_count(), 2);
        assert!(matches!(kernel.simd_lanes(), 1 | 2 | 4));

        for &len in LENGTHS {
            let mut x_re = guarded_plane(len, |i| i as f64 * 0.25 - 1.0);
            let mut x_im = guarded_plane(len, |i| 0.5 + i as f64 * 0.0625);
            let mut y_re = guarded_plane(len, |i| 2.0 - i as f64 * 0.125);
            let mut y_im = guarded_plane(len, |i| -0.75 + i as f64 * 0.03125);
            let mut out_re = guarded_plane(len, |_| f64::NAN);
            let mut out_im = guarded_plane(len, |_| f64::NAN);
            {
                let mut bound = kernel
                    .bind(vec![
                        &mut x_re[1..len + 1],
                        &mut x_im[1..len + 1],
                        &mut y_re[1..len + 1],
                        &mut y_im[1..len + 1],
                        &mut out_re[1..len + 1],
                        &mut out_im[1..len + 1],
                    ])
                    .unwrap();
                bound.execute_all().unwrap();
            }
            for index in 0..len {
                let xr = x_re[index + 1];
                let xi = x_im[index + 1];
                assert_eq!(
                    out_re[index + 1],
                    xr * xr - xi * xi + y_re[index + 1],
                    "length {len}, point {index}, real"
                );
                assert_eq!(
                    out_im[index + 1],
                    2.0 * xr * xi + y_im[index + 1],
                    "length {len}, point {index}, imaginary"
                );
            }
            for plane in [&x_re, &x_im, &y_re, &y_im, &out_re, &out_im] {
                assert_guards(plane);
            }
        }
    }

    #[test]
    fn unaligned_subrange_uses_scalar_head_and_tail() {
        let bytes = application_bytes(false, 2);
        let kernel = SymjitPlaneKernel::load_bytes(
            &bytes,
            "unaligned-contract.symjit",
            SymjitPlaneLayout::real(2, 1),
        )
        .unwrap();
        let len = 19;
        let mut x: Vec<f64> = (0..len).map(|i| i as f64 + 1.0).collect();
        let mut y: Vec<f64> = (0..len).map(|i| i as f64 * 0.5).collect();
        let mut out = vec![SENTINEL; len];
        {
            let mut bound = kernel
                .bind(vec![x.as_mut_slice(), y.as_mut_slice(), out.as_mut_slice()])
                .unwrap();
            bound.execute_range(1, len - 3).unwrap();
        }
        assert_eq!(out[0], SENTINEL);
        assert_eq!(out[len - 2], SENTINEL);
        assert_eq!(out[len - 1], SENTINEL);
        for index in 1..len - 2 {
            assert_eq!(out[index], x[index] * x[index] + y[index]);
        }
    }

    #[test]
    fn raw_binding_executes_duplicate_inputs_without_overlapping_references() {
        let bytes = application_bytes(false, 2);
        let kernel = SymjitPlaneKernel::load_bytes(
            &bytes,
            "duplicate-contract.symjit",
            SymjitPlaneLayout::real(2, 1),
        )
        .unwrap();
        let mut input: Vec<f64> = (1..=8).map(f64::from).collect();
        let mut output = vec![0.0; 8];
        let duplicate = input.as_mut_ptr();
        {
            let mut bound = kernel
                .bind_raw(vec![
                    // SAFETY: Both descriptors share live, synchronized
                    // read-only input storage for the duration of execution.
                    unsafe { PlaneDescriptor::from_raw_parts(duplicate, input.len()).unwrap() },
                    unsafe { PlaneDescriptor::from_raw_parts(duplicate, input.len()).unwrap() },
                    PlaneDescriptor::from_slice(output.as_mut_slice()),
                ])
                .unwrap();
            bound.execute_all().unwrap();
        }
        for (index, value) in input.iter().copied().enumerate() {
            assert_eq!(output[index], value * value + value);
        }
    }

    #[test]
    fn raw_binding_executes_input_output_alias_from_prologue_snapshot() {
        let bytes = application_bytes(false, 2);
        let kernel = SymjitPlaneKernel::load_bytes(
            &bytes,
            "alias-contract.symjit",
            SymjitPlaneLayout::real(2, 1),
        )
        .unwrap();
        let mut first: Vec<f64> = (1..=8).map(f64::from).collect();
        let original_first = first.clone();
        let mut second: Vec<f64> = (0..8).map(|index| 10.0 + index as f64).collect();
        let alias = first.as_mut_ptr();
        {
            let mut bound = kernel
                .bind_raw(vec![
                    // SAFETY: The P-kernel snapshots its inputs in the
                    // prologue before overwriting the aliased output.
                    unsafe { PlaneDescriptor::from_raw_parts(alias, first.len()).unwrap() },
                    PlaneDescriptor::from_slice(second.as_mut_slice()),
                    unsafe { PlaneDescriptor::from_raw_parts(alias, first.len()).unwrap() },
                ])
                .unwrap();
            bound.execute_all().unwrap();
        }
        for index in 0..first.len() {
            assert_eq!(
                first[index],
                original_first[index] * original_first[index] + second[index]
            );
        }
    }

    #[test]
    fn disjoint_raw_binding_executes_without_descriptor_repacking() {
        let bytes = application_bytes(false, 2);
        let kernel = SymjitPlaneKernel::load_bytes(
            &bytes,
            "raw-contract.symjit",
            SymjitPlaneLayout::real(2, 1),
        )
        .unwrap();
        let mut x = vec![2.0, 3.0, 4.0, 5.0];
        let mut y = vec![1.0, 2.0, 3.0, 4.0];
        let mut out = vec![0.0; 4];
        {
            let mut table = kernel
                .build_raw_table(vec![
                    PlaneDescriptor::from_slice(x.as_mut_slice()),
                    PlaneDescriptor::from_slice(y.as_mut_slice()),
                    PlaneDescriptor::from_slice(out.as_mut_slice()),
                ])
                .unwrap();
            assert_eq!(table.point_count(), 4);
            let table_address = table.as_symjit_ptr().addr();
            kernel.execute_table(&mut table, 0, 4).unwrap();
            assert_eq!(table_address, table.as_symjit_ptr().addr());
        }
        assert_eq!(out, [5.0, 11.0, 19.0, 29.0]);
    }

    #[test]
    fn raw_table_rebind_preserves_descriptor_allocation_and_executes_new_planes() {
        let bytes = application_bytes(false, 2);
        let kernel = SymjitPlaneKernel::load_bytes(
            &bytes,
            "raw-rebind-contract.symjit",
            SymjitPlaneLayout::real(2, 1),
        )
        .unwrap();
        let mut first_x = vec![2.0, 3.0, 4.0, 5.0];
        let mut first_y = vec![1.0, 2.0, 3.0, 4.0];
        let mut first_out = vec![0.0; 4];
        let mut second_x = vec![6.0, 7.0, 8.0, 9.0];
        let mut second_y = vec![5.0, 6.0, 7.0, 8.0];
        let mut second_out = vec![0.0; 4];

        {
            let mut table = kernel
                .build_raw_table(vec![
                    PlaneDescriptor::from_slice(first_x.as_mut_slice()),
                    PlaneDescriptor::from_slice(first_y.as_mut_slice()),
                    PlaneDescriptor::from_slice(first_out.as_mut_slice()),
                ])
                .unwrap();
            let table_address = table.as_symjit_ptr().addr();
            kernel.execute_table(&mut table, 0, 4).unwrap();

            let replacement = [
                // SAFETY: Each allocation is live, aligned, synchronized, and
                // retains its length until the table is dropped.
                unsafe {
                    PlaneDescriptor::from_raw_parts(second_x.as_mut_ptr(), second_x.len()).unwrap()
                },
                unsafe {
                    PlaneDescriptor::from_raw_parts(second_y.as_mut_ptr(), second_y.len()).unwrap()
                },
                unsafe {
                    PlaneDescriptor::from_raw_parts(second_out.as_mut_ptr(), second_out.len())
                        .unwrap()
                },
            ];
            // SAFETY: No kernel invocation overlaps the rebind and all
            // replacement allocations outlive the table.
            unsafe { table.rebind_raw(&replacement) }.unwrap();
            assert_eq!(table_address, table.as_symjit_ptr().addr());
            kernel.execute_table(&mut table, 0, 4).unwrap();
            assert_eq!(table_address, table.as_symjit_ptr().addr());
        }

        assert_eq!(first_out, [5.0, 11.0, 19.0, 29.0]);
        assert_eq!(second_out, [41.0, 55.0, 71.0, 89.0]);
    }

    fn status_kernel(
        _mem: *const f64,
        _states: *const SymjitPlaneDescriptor<f64>,
        index: usize,
        _params: *const f64,
    ) -> i32 {
        if index == 2 { 17 } else { 0 }
    }

    #[test]
    fn nonzero_kernel_status_is_contained() {
        let failure = invoke_range(
            status_kernel,
            None,
            1,
            std::ptr::null(),
            std::ptr::null(),
            0,
            4,
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            failure,
            StatusFailure {
                kind: CallKind::Scalar,
                index: 2,
                status: 17,
            }
        );
    }

    #[test]
    fn symjit_panics_are_contained_with_dependency_context() {
        let error = guard_symjit_panic(
            || panic!("probe panic"),
            || "exercise the panic-containment probe".to_string(),
        )
        .unwrap_err();
        assert_eq!(error.kind(), RusticolErrorKind::Compatibility);
        assert!(
            error
                .message()
                .contains("SymJIT panicked while trying to exercise the panic-containment probe")
        );
        assert!(error.message().contains("probe panic"));
    }

    #[test]
    fn storage_header_dimensions_and_trailing_bytes_fail_closed() {
        let bytes = application_bytes(false, 2);

        let mut bad_magic = bytes.clone();
        bad_magic[0] ^= 1;
        let error = SymjitPlaneKernel::load_bytes(
            &bad_magic,
            "bad-magic.symjit",
            SymjitPlaneLayout::real(2, 1),
        )
        .unwrap_err();
        assert_eq!(error.kind(), RusticolErrorKind::Compatibility);

        let error = SymjitPlaneKernel::load_bytes(
            &bytes,
            "bad-shape.symjit",
            SymjitPlaneLayout::real(3, 1),
        )
        .unwrap_err();
        assert_eq!(error.kind(), RusticolErrorKind::Integrity);

        let mut trailing = bytes;
        trailing.push(0);
        let error = SymjitPlaneKernel::load_bytes(
            &trailing,
            "trailing.symjit",
            SymjitPlaneLayout::real(2, 1),
        )
        .unwrap_err();
        assert_eq!(error.kind(), RusticolErrorKind::Integrity);
    }

    #[test]
    fn migration_contract_rejects_stored_simd512_configuration() {
        let mut config = explicit_plane_config(false, 2, true).unwrap();
        config.enable_simd512(true);
        let bytes = application_bytes_with_config(config);
        let error =
            SymjitPlaneKernel::load_bytes(&bytes, "simd512.symjit", SymjitPlaneLayout::real(2, 1))
                .unwrap_err();
        assert_eq!(error.kind(), RusticolErrorKind::Compatibility);
        assert!(error.message().contains("SIMD512"));
    }

    #[test]
    fn migration_contract_accepts_only_scalar_f64x2_or_f64x4_runtime_lanes() {
        let path = Path::new("lane-contract.symjit");
        for (has_simd, lanes) in [(false, 1), (true, 2), (true, 4)] {
            validate_simd_lane_width(has_simd, lanes, path).unwrap();
        }
        for (has_simd, lanes) in [(true, 1), (false, 2), (true, 8)] {
            let error = validate_simd_lane_width(has_simd, lanes, path).unwrap_err();
            assert_eq!(error.kind(), RusticolErrorKind::Compatibility);
            assert!(error.message().contains("unsupported SIMD lane width"));
        }
    }

    #[test]
    fn compile_helper_rejects_invalid_cold_path_inputs() {
        let error =
            compile_symbolica_program_to_plane_application_bytes("", 2, 1, 2, false).unwrap_err();
        assert_eq!(error.kind(), RusticolErrorKind::InvalidArgument);
        let error = compile_symbolica_program_to_plane_application_bytes(MODEL, 2, 1, 4, false)
            .unwrap_err();
        assert_eq!(error.kind(), RusticolErrorKind::InvalidArgument);
    }
}
