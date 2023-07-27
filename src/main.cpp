#include <torch/torch.h>
#include <iostream>
#include <vector>

#include "stp/stp.hpp"

int main()
{
  STP puzzle = STP(2, 2);

  int state[4] = {0, 1, 2, 3};
  puzzle.initState(state);

  std::vector<int> pattern = std::vector<int>({0, 2});

  std::cout << puzzle.hashState(pattern) << std::endl;
}
