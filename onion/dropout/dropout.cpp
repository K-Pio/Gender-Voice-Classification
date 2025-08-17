#include <torch/extension.h>

#ifdef WITH_CUDA
std::vector<torch::Tensor> dropout_forward_cuda(
    torch::Tensor x, float p, bool training);
torch::Tensor dropout_backward_cuda(
    torch::Tensor grad_y, torch::Tensor mask, float p);
#endif

class DropoutFunction : public torch::autograd::Function<DropoutFunction> {
public:
    static torch::Tensor forward(torch::autograd::AutogradContext* ctx,
                                 torch::Tensor x, double p, bool training) 
    {
        ctx->saved_data["p"] = p;
        if ( !training ) 
        {
            ctx->saved_data["mask"] = torch::Tensor();
            return x;
        }
#ifdef  WITH_CUDA
        auto out = dropout_forward_cuda(x.contiguous(), p, training);
        ctx->save_for_backward({out[1]}); // maska
        return out[0];
#else
        TORCH_CHECK(false, "CUDA only in this stub");
#endif
    }

    static torch::autograd::tensor_list backward(torch::autograd::AutogradContext* ctx,
                                                 torch::autograd::tensor_list grad_outputs) 
    {
        auto p = ctx->saved_data["p"].toDouble();
        auto saved = ctx->get_saved_variables();
        auto mask = saved.size() ? saved[0] : torch::Tensor();
#ifdef  WITH_CUDA
        auto grad_x = dropout_backward_cuda(grad_outputs[0].contiguous(), mask, p);
        return {grad_x, torch::Tensor(), torch::Tensor()};
#else
        TORCH_CHECK(false, "CUDA only in this stub");
#endif
    }
};

torch::Tensor custom_dropout(torch::Tensor x, double p, bool training) 
{
    return DropoutFunction::apply(x, p, training);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) 
{
    m.def("custom_dropout", &custom_dropout,
          "Custom Dropout (CUDA)",
          py::arg("input"),
          py::arg("p") = 0.5,
          py::arg("training") = true);
}
