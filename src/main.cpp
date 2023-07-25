#include <torch/torch.h>
#include <iostream>

#include "model/model.hpp"

int main()
{
  torch::Tensor tensor = torch::rand({2, 3});
  std::cout << tensor << std::endl;

  QNT_5_5 *model = new QNT_5_5();
}
