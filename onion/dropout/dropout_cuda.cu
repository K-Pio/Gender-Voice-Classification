#include <torch/extension.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

template <typename scalar_t>
__global__ void dropout_forward_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ y,
    uint8_t* __restrict__ mask,
    float p, uint64_t seed, uint64_t offset, int64_t N)
{
    int linear_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_tid = blockDim.x * gridDim.x;

    curandStatePhilox4_32_10_t rng;
    curand_init(seed, /*subsequence=*/offset + linear_tid, /*offset=*/0, &rng);

    const float scale = (p < 1.f) ? 1.f / (1.f - p) : 0.f;
    using acc_t = at::acc_type<scalar_t, /*is_cuda=*/true>;

    for ( int64_t base = int64_t(linear_tid) * 4; base < N; base += int64_t(stride_tid) * 4 )
    {
        float4 r4 = curand_uniform4(&rng);
        float r[4] = {r4.x, r4.y, r4.z, r4.w};

        #pragma unroll
        for ( int k = 0; k < 4; ++k )
        {
            int64_t i = base + k;
            if ( i >= N ) break;

            uint8_t m = static_cast<uint8_t>(r[k] > p);
            mask[i] = m;

            acc_t xi = static_cast<acc_t>(x[i]);
            acc_t yi = m ? static_cast<acc_t>(scale) * xi : static_cast<acc_t>(0.0f);
            y[i] = static_cast<scalar_t>(yi);
        }
    }
}

template <typename scalar_t>
__global__ void dropout_backward_kernel(
    const scalar_t* __restrict__ grad_y,
    const uint8_t* __restrict__ mask,
    scalar_t* __restrict__ grad_x,
    float p, int64_t N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= N ) return;

    float scale = (p < 1.f) ? 1.0f / (1.0f - p) : 0.0f;

    using acc_t = at::acc_type<scalar_t, /*is_cuda=*/true>;
    acc_t gy = static_cast<acc_t>(grad_y[idx]);
    acc_t gx = mask[idx] ? static_cast<acc_t>(scale) * gy : static_cast<acc_t>(0.0f);
    grad_x[idx] = static_cast<scalar_t>(gx);
}

std::vector<torch::Tensor> dropout_forward_cuda(
    torch::Tensor x, float p, bool training)
{
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(p >= 0.f && p < 1.f, "p must be in [0,1)");
    auto x_contig = x.contiguous();
    auto N = x_contig.numel();

    if ( !training || p == 0.f )
    {
        auto y = x_contig.clone();
        auto mask = torch::ones_like(x_contig, x_contig.options().dtype(torch::kUInt8));
        return {y, mask};
    }

    auto y = torch::empty_like(x_contig);
    auto mask = torch::empty(x_contig.sizes(), x_contig.options().dtype(torch::kUInt8));

    auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
        c10::nullopt, at::cuda::detail::getDefaultCUDAGenerator()
    );
    auto engine_inputs = gen->philox_engine_inputs((N + 3) / 4);
    uint64_t seed = engine_inputs.first;
    uint64_t offset = engine_inputs.second;

    constexpr int THREADS = 256;
    int blocks = static_cast<int>((N + THREADS - 1) / THREADS);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x_contig.scalar_type(), "dropout_forward_cuda", [&]
    {
        dropout_forward_kernel<scalar_t><<<blocks, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
            x_contig.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>(),
            mask.data_ptr<uint8_t>(),
            p,
            seed,
            offset,
            N);
    });

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {y, mask};
}

torch::Tensor dropout_backward_cuda(
    torch::Tensor grad_y, torch::Tensor mask, float p)
{
    TORCH_CHECK(grad_y.is_cuda() && mask.is_cuda(), "CUDA tensors expected");
    TORCH_CHECK(p >= 0.f && p < 1.f, "p must be in [0,1)");
    auto gy = grad_y.contiguous();
    auto m = mask.contiguous();
    auto N = gy.numel();

    auto grad_x = torch::empty_like(gy);

    constexpr int THREADS = 256;
    int blocks = static_cast<int>((N + THREADS - 1) / THREADS);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, gy.scalar_type(), "dropout_backward_cuda", [&]
    {
        dropout_backward_kernel<scalar_t><<<blocks, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
            gy.data_ptr<scalar_t>(),
            m.data_ptr<uint8_t>(),
            grad_x.data_ptr<scalar_t>(),
            p,
            N);
    });

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grad_x;
}
