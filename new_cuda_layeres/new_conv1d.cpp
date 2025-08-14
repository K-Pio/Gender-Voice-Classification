#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#ifdef WITH_CUDA
std::vector<torch::Tensor> conv1d_forward_cuda(
    torch::Tensor x, torch::Tensor w, c10::optional<torch::Tensor> b,
    int stride, int pad, int dil);
std::vector<torch::Tensor> conv1d_backward_cuda(
    torch::Tensor x, torch::Tensor w, c10::optional<torch::Tensor> b,
    torch::Tensor grad_y, int stride, int pad, int dil);
#endif

// CPU fallback
static std::vector<torch::Tensor> conv1d_forward_cpu(
    torch::Tensor x, torch::Tensor w, c10::optional<torch::Tensor> b,
    int stride, int pad, int dil)
{
    auto y = torch::conv1d(x, w, b.has_value() ? b.value() : torch::Tensor(),
                           stride, pad, dil);
    return {y};
}

static std::vector<torch::Tensor> conv1d_backward_cpu(
    torch::Tensor x, torch::Tensor w, c10::optional<torch::Tensor> b,
    torch::Tensor grad_y, int stride, int pad, int dil)
{
    x.set_requires_grad(true);
    w.set_requires_grad(true);
    torch::Tensor bias = b.has_value() ? b.value() : torch::Tensor();
    if (b.has_value()) {
        bias.set_requires_grad(true);
    }

    // Forward pass
    auto y = torch::conv1d(x, w, bias, stride, pad, dil);

    // Backward pass
    y.backward(grad_y, /*keep_graph=*/true);

    // get gradients
    torch::Tensor grad_x = x.grad();
    torch::Tensor grad_w = w.grad();
    torch::Tensor grad_b = b.has_value() ? bias.grad() : torch::Tensor();

    return {grad_x, grad_w, grad_b};
}


// Autograd Function bridge
class Conv1dFunction : public torch::autograd::Function<Conv1dFunction> {
public:
    static torch::Tensor forward(torch::autograd::AutogradContext* ctx,
                                 torch::Tensor x, torch::Tensor w, c10::optional<torch::Tensor> b,
                                 int64_t stride, int64_t pad, int64_t dil) {
        ctx->save_for_backward({x, w, b.has_value() ? b.value() : torch::Tensor()});
        ctx->saved_data["has_bias"] = b.has_value();
        ctx->saved_data["stride"] = stride;
        ctx->saved_data["pad"] = pad;
        ctx->saved_data["dil"] = dil;

        if (x.is_cuda()) {
#ifdef WITH_CUDA
            auto out = conv1d_forward_cuda(x.contiguous(), w.contiguous(), b, (int)stride, (int)pad, (int)dil);
            return out[0];
#else
            TORCH_CHECK(false, "CUDA not compiled in this extension");
#endif
        } else {
            auto out = conv1d_forward_cpu(x.contiguous(), w.contiguous(), b, (int)stride, (int)pad, (int)dil);
            return out[0];
        }
    }

    static torch::autograd::tensor_list backward(torch::autograd::AutogradContext* ctx,
                                                 torch::autograd::tensor_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto x = saved[0];
        auto w = saved[1];
        auto b = saved[2].defined() ? c10::make_optional(saved[2]) : c10::optional<torch::Tensor>();

        int stride = ctx->saved_data["stride"].toInt();
        int pad    = ctx->saved_data["pad"].toInt();
        int dil    = ctx->saved_data["dil"].toInt();

        torch::Tensor grad_y = grad_outputs[0].contiguous();

        torch::Tensor grad_x, grad_w, grad_b;
        if (x.is_cuda()) {
#ifdef WITH_CUDA
            auto grads = conv1d_backward_cuda(x, w, b, grad_y, stride, pad, dil);
            grad_x = grads[0]; grad_w = grads[1]; grad_b = grads[2];
#else
            TORCH_CHECK(false, "CUDA not compiled in this extension");
#endif
        } else {
            auto grads = conv1d_backward_cpu(x, w, b, grad_y, stride, pad, dil);
            grad_x = grads[0]; grad_w = grads[1]; grad_b = grads[2];
        }

        // x, w, b, stride, pad, dil
        return {
            grad_x, grad_w, b.has_value() ? grad_b : torch::Tensor(),
            torch::Tensor(), torch::Tensor(), torch::Tensor()
        };
    }
};

torch::Tensor custom_conv1d(torch::Tensor x, torch::Tensor w, c10::optional<torch::Tensor> b,
                            int64_t stride, int64_t pad, int64_t dil) {
    return Conv1dFunction::apply(x, w, b, stride, pad, dil);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_conv1d", &custom_conv1d,
          "Custom Conv1d (autograd, CPU/CUDA)",
          pybind11::arg("input"),
          pybind11::arg("weight"),
          pybind11::arg("bias") = c10::optional<torch::Tensor>(),
          pybind11::arg("stride") = 1,
          pybind11::arg("padding") = 0,
          pybind11::arg("dilation") = 1);
}
