//#![allow(dead_code, unused)]
use image::{ImageBuffer, Rgba};
use show_image::create_window;
use std::time::Instant;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, CopyImageToBufferInfo,
};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::physical::PhysicalDevice;
use vulkano::device::{Device, DeviceCreateInfo, QueueCreateInfo};
use vulkano::format::Format;
use vulkano::image::{ImageDimensions, StorageImage};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::pipeline::ComputePipeline;
use vulkano::pipeline::Pipeline;
use vulkano::pipeline::PipelineBindPoint;
use vulkano::sync::{self, GpuFuture};

// MandelBrot shader
mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        src: "#version 460

        layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
        
        layout(set = 0, binding = 0, rgba8) uniform writeonly image2D img;
        
        void main() {
            vec2 norm_coordinates = (gl_GlobalInvocationID.xy + vec2(0.5)) / vec2(imageSize(img));
            vec2 c = (norm_coordinates - vec2(0.5)) * 2.0 - vec2(1.0, 0.0);
        
            vec2 z = vec2(0.0, 0.0);
            float i;
            float samples = 200.0;
            for (i = 0.0; i < 1.0; i += 1.0/samples) {
                z = vec2(
                    z.x * z.x - z.y * z.y + c.x,
                    z.y * z.x + z.x * z.y + c.y
                );
        
                if (length(z) > 4.0) {
                    break;
                }
            }
        
            vec4 to_write = vec4(vec3(i), 1.0);
            imageStore(img, ivec2(gl_GlobalInvocationID.xy), to_write);
        }"
    }
}

// Disk Shader
mod cs2 {
    vulkano_shaders::shader! {
        ty: "compute",
        src: "#version 460

        layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
        
        layout(set = 0, binding = 0, rgba8) uniform writeonly image2D img;
        
        float disk(vec2 UV, float radius, float smoothing) {
            float x = UV.x;
            float y = UV.y;
            float disk = x*x + y*y;
            return smoothstep(disk, disk+smoothing, radius*radius);
        }

        void main() {
            vec2 UV = (gl_GlobalInvocationID.xy + vec2(0.5)) / vec2(imageSize(img));
            UV -= vec2(0.5);

            float c = disk(UV, 0.4, 0.01) - disk(UV, 0.2, 0.02);
            vec4 to_write = vec4(vec3(c), 1.0);
            imageStore(img, ivec2(gl_GlobalInvocationID.xy), to_write);
        }"
    }
}

#[show_image::main]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let resolution = std::env::args().skip(1).next().unwrap().parse().unwrap();

    // Create application's entry to Vulkan API
    let instance = Instance::new(InstanceCreateInfo::default()).unwrap();

    // List all the physical devices that supports Vulkan
    let physical_device = PhysicalDevice::enumerate(&instance).next().unwrap();

    // Find and select the first queue family (threads group) that supports graphics and compute
    let queue_family = physical_device
        .queue_families()
        .find(|&queue_family| queue_family.supports_compute())
        .unwrap();

    // Create vulkan context from the physical device using the selected queue family.
    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo::family(queue_family)],
            ..Default::default()
        },
    )
    .unwrap();

    // Selecting the first queue from the selected queue family
    let queue = queues.next().unwrap();

    let image = StorageImage::new(
        device.clone(),
        ImageDimensions::Dim2d {
            width: resolution,
            height: resolution,
            array_layers: 1,
        },
        Format::R8G8B8A8_UNORM,
        Some(queue.family()),
    )
    .unwrap();

    let shader = cs2::load(device.clone()).unwrap();

    let compute_pipeline = ComputePipeline::new(
        device.clone(),
        shader.entry_point("main").unwrap(),
        &(),
        None,
        |_| {},
    )
    .unwrap();

    let view = ImageView::new_default(image.clone()).unwrap();

    use vulkano::image::view::ImageView;
    let layout = compute_pipeline.layout().set_layouts().get(0).unwrap();
    let set = PersistentDescriptorSet::new(
        layout.clone(),
        [WriteDescriptorSet::image_view(0, view.clone())], // 0 is the binding
    )
    .unwrap();

    let buf = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::all(),
        false,
        (0..resolution * resolution * 4).map(|_| 0u8),
    )
    .expect("failed to create buffer");

    let mut builder = AutoCommandBufferBuilder::primary(
        device.clone(),
        queue.family(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();
    builder
        .bind_pipeline_compute(compute_pipeline.clone())
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            compute_pipeline.layout().clone(),
            0,
            set,
        )
        .dispatch([resolution / 8, resolution / 8, 1])
        .unwrap()
        .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
            image.clone(),
            buf.clone(),
        ))
        .unwrap();

    let command_buffer = builder.build().unwrap();

    let future = sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();

    let start = Instant::now();
    future.wait(None).unwrap();
    let gpu_time = Instant::now()-start;

    println!("The GPU took {:?}", gpu_time);

    let buffer_content = buf.read().unwrap();
    let image = ImageBuffer::<Rgba<u8>, _>::from_raw(resolution, resolution, &buffer_content[..]).unwrap();
    
    let window = create_window("image", Default::default())?;
    window.set_image("image-001", image)?;
    window.wait_until_destroyed().unwrap();

    println!("Everything succeeded!");
    Ok(())
}
