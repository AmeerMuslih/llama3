#include <torch/extension.h>

#include <vector>
#include "smt_sa_os.cpp"
#include <cstdint>
// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CPU(x) AT_ASSERTM(x.type().is_cpu(), #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CPU(x); CHECK_CONTIGUOUS(x)


std::vector<torch::Tensor> sa_main(
    torch::Tensor a, torch::Tensor b,
    unsigned int dim, unsigned int t, unsigned int buf_size) {
    smt_sa_os<int32_t> sa = smt_sa_os<int32_t>(dim, t, buf_size);
	torch::Tensor a_tmp = a.to(torch::kInt32);
	torch::Tensor b_tmp = b.to(torch::kInt32);
    sa.set_inputs(a_tmp, b_tmp);

    return sa.go();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("exec", &sa_main, "Start SMT-SA");
}

