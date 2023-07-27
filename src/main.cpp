#include <torch/torch.h>
#include <iostream>

#include "stp/stp.hpp"

int main()
{
  STP puzzle = STP(2, 2);

  int state[4] = {0, 1, 2, 3};
  puzzle.initState(state);

  std::cout << puzzle.hashState() << std::endl;
}
