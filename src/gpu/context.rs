//! GPU device, adapter, and queue initialization.
//!
//! Handles wgpu instance creation, adapter selection (preferring high-performance
//! discrete GPUs), and device/queue acquisition. Logs the selected backend
//! (Metal on macOS, Vulkan on Linux) for diagnostics.

use wgpu;

/// Holds the wgpu device, queue, and adapter metadata.
///
/// This is the entry point for all GPU operations. Create once at startup
/// and pass to `GpuGemm` for compute pipeline execution.
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter_name: String,
    pub backend: String,
}

impl GpuContext {
    /// Initialize the GPU context.
    ///
    /// Requests a high-performance adapter (discrete GPU preferred over integrated).
    /// Uses `pollster::block_on` for synchronous initialization.
    ///
    /// # Panics
    /// Panics if no suitable GPU adapter is found.
    pub fn new() -> Self {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Self {
        // Create wgpu instance with default backends (Metal on macOS, Vulkan on Linux)
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // Request a high-performance adapter
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .expect("Failed to find a suitable GPU adapter");

        let info = adapter.get_info();
        let adapter_name = info.name.clone();
        let backend = format!("{:?}", info.backend);

        eprintln!(
            "[Quasar-BLAS GPU] Adapter: {} | Backend: {} | Device Type: {:?}",
            adapter_name, backend, info.device_type
        );

        // Request device with default limits
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("quasar-blas-device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .expect("Failed to create GPU device");

        Self {
            device,
            queue,
            adapter_name,
            backend,
        }
    }
}
