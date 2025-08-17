#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void conv1d_forward_kernel(
    const scalar_t* __restrict__ x,      // [B, C_in, T_in]
    const scalar_t* __restrict__ w,      // [C_out, C_in, K]
    const scalar_t* __restrict__ b,      // [C_out] (opcjonalnie)
    scalar_t* __restrict__ y,            // [B, C_out, T_out]
    int B, int C_in, int T_in,
    int C_out, int K,
    int stride, int pad, int dil,
    int T_out)
{
    int b_idx  = blockIdx.x;     // batch
    int co     = blockIdx.y;     // out channel
    int t_out  = blockIdx.z * blockDim.x + threadIdx.x; // time index

    if ( b_idx >= B || co >= C_out || t_out >= T_out ) return;

    scalar_t acc = b ? b[co] : static_cast<scalar_t>(0);

    int t_in0 = t_out * stride - pad;

    for ( int ci = 0; ci < C_in; ++ci )
    {
        const scalar_t* x_ptr = x + ((b_idx * C_in + ci) * T_in);
        const scalar_t* w_ptr = w + ((co * C_in + ci) * K);
        #pragma unroll
        for ( int k = 0; k < K; ++k )
        {
            int t_in = t_in0 + k * dil;
            if ( 0 <= t_in && t_in < T_in )
            {
                acc += x_ptr[t_in] * w_ptr[k];
            }
        }
    }
    y[((b_idx * C_out + co) * T_out) + t_out] = acc;
}

template <typename scalar_t>
__global__ void conv1d_grad_input_kernel(
    const scalar_t* __restrict__ grad_y,  // [B, C_out, T_out]
    const scalar_t* __restrict__ w,       // [C_out, C_in, K]
    scalar_t* __restrict__ grad_x,        // [B, C_in, T_in]
    int B, int C_in, int T_in,
    int C_out, int K,
    int stride, int pad, int dil,
    int T_out)
{
    int b_idx = blockIdx.x;
    int ci    = blockIdx.y;
    int t_in  = blockIdx.z * blockDim.x + threadIdx.x;
    if ( b_idx >= B || ci >= C_in || t_in >= T_in ) return;

    scalar_t acc = 0;
    for ( int co = 0; co < C_out; ++co )
    {
        const scalar_t* w_ptr = w + ((co * C_in + ci) * K);
        const scalar_t* gy_ptr = grad_y + ((b_idx * C_out + co) * T_out);
        
        #pragma unroll
        for ( int k = 0; k < K; ++k )
        {
            int num = t_in + pad - k * dil;
            if ( num % stride == 0 )
            {
                int t_out = num / stride;
                if ( 0 <= t_out && t_out < T_out )
                {
                    acc += gy_ptr[t_out] * w_ptr[k];
                }
            }
        }
    }
    grad_x[((b_idx * C_in + ci) * T_in) + t_in] = acc;
}

template <typename scalar_t>
__global__ void conv1d_grad_weight_kernel(
    const scalar_t* __restrict__ x,       // [B, C_in, T_in]
    const scalar_t* __restrict__ grad_y,  // [B, C_out, T_out]
    scalar_t* __restrict__ grad_w,        // [C_out, C_in, K]
    int B, int C_in, int T_in,
    int C_out, int K,
    int stride, int pad, int dil,
    int T_out)
{
    int co = blockIdx.x;
    int ci = blockIdx.y;
    int k  = blockIdx.z * blockDim.x + threadIdx.x;
    if ( co >= C_out || ci >= C_in || k >= K ) return;

    scalar_t acc = 0;
    const scalar_t* w_grad_ptr = grad_w + ((co * C_in + ci) * K + k);

    for ( int b = 0; b < B; ++b )
    {
        const scalar_t* x_ptr  = x  + ((b * C_in + ci) * T_in);
        const scalar_t* gy_ptr = grad_y + ((b * C_out + co) * T_out);
        // t_in = t_out*stride - pad + k*dil
        for ( int t_out = 0; t_out < T_out; ++t_out )
        {
            int t_in = t_out * stride - pad + k * dil;
            if ( 0 <= t_in && t_in < T_in )
            {
                acc += gy_ptr[t_out] * x_ptr[t_in];
            }
        }
    }
    grad_w[((co * C_in + ci) * K) + k] = acc;
}

template <typename scalar_t>
__global__ void conv1d_grad_bias_kernel(
    const scalar_t* __restrict__ grad_y, // [B, C_out, T_out]
    scalar_t* __restrict__ grad_b,       // [C_out]
    int B, int C_out, int T_out)
{
    int co = blockIdx.x * blockDim.x + threadIdx.x;
    if ( co >= C_out ) return;
    scalar_t acc = 0;
    for ( int b = 0; b < B; ++b )
    {
        const scalar_t* gy_ptr = grad_y + ((b * C_out + co) * T_out);
        for ( int t = 0; t < T_out; ++t )
        {
            acc += gy_ptr[t];
        }
    }
    grad_b[co] = acc;
}

std::vector<torch::Tensor> conv1d_forward_cuda(
    torch::Tensor x, torch::Tensor w, c10::optional<torch::Tensor> b,
    int stride, int pad, int dil)
{
    const auto B = x.size(0);
    const auto C_in = x.size(1);
    const auto T_in = x.size(2);
    const auto C_out = w.size(0);
    const auto K = w.size(2);
    const auto T_out = (T_in + 2*pad - dil*(K-1) - 1) / stride + 1;

    auto y = torch::empty({B, C_out, T_out}, x.options());

    dim3 threads(256);
    dim3 blocks(B, C_out, (T_out + threads.x - 1) / threads.x);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "conv1d_forward_cuda", [&] {
        const scalar_t* b_ptr = b.has_value() ? b.value().data_ptr<scalar_t>() : nullptr;
        conv1d_forward_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(), w.data_ptr<scalar_t>(), b_ptr, y.data_ptr<scalar_t>(),
            B, C_in, T_in, C_out, K, stride, pad, dil, T_out
        );
    });

    return {y};
}

std::vector<torch::Tensor> conv1d_backward_cuda(
    torch::Tensor x, torch::Tensor w, c10::optional<torch::Tensor> b,
    torch::Tensor grad_y, int stride, int pad, int dil)
{
    const auto B = x.size(0);
    const auto C_in = x.size(1);
    const auto T_in = x.size(2);
    const auto C_out = w.size(0);
    const auto K = w.size(2);
    const auto T_out = grad_y.size(2);

    auto grad_x = torch::zeros_like(x);
    auto grad_w = torch::zeros_like(w);
    torch::Tensor grad_b = torch::Tensor();

    dim3 threads(256);

    {
        dim3 blocks(B, C_in, (T_in + threads.x - 1) / threads.x);
        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "conv1d_grad_input_cuda", [&] {
            conv1d_grad_input_kernel<scalar_t><<<blocks, threads>>>(
                grad_y.data_ptr<scalar_t>(), w.data_ptr<scalar_t>(), grad_x.data_ptr<scalar_t>(),
                B, C_in, T_in, C_out, K, stride, pad, dil, T_out
            );
        });
    }

    {
        dim3 blocks(C_out, C_in, (K + threads.x - 1) / threads.x);
        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "conv1d_grad_weight_cuda", [&] {
            conv1d_grad_weight_kernel<scalar_t><<<blocks, threads>>>(
                x.data_ptr<scalar_t>(), grad_y.data_ptr<scalar_t>(), grad_w.data_ptr<scalar_t>(),
                B, C_in, T_in, C_out, K, stride, pad, dil, T_out
            );
        });
    }

    if ( b.has_value() )
    {
        grad_b = torch::empty_like(b.value());
        dim3 blocks((C_out + 255) / 256);
        dim3 threads_b(256);
        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "conv1d_grad_bias_cuda", [&] {
            conv1d_grad_bias_kernel<scalar_t><<<blocks, threads_b>>>(
                grad_y.data_ptr<scalar_t>(), grad_b.data_ptr<scalar_t>(), B, C_out, T_out
            );
        });
    }

    return {grad_x, grad_w, grad_b};
}
