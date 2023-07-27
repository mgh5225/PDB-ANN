#include <torch/torch.h>
#include <iostream>
#include <vector>

#include "stp/stp.hpp"

int main()
{
  STP puzzle = STP(3, 3);

  std::vector<int> state = std::vector<int>({3, 0, 1, 2, 4, 6, 7, 8, 5});
  std::vector<int> pattern = std::vector<int>({0, 1, 6, 5});
  puzzle.initState(state);
  puzzle.toAbstract(pattern);

  std::cout << puzzle.getState() << std::endl;

  puzzle.move(STPAction::RIGHT);
  puzzle.move(STPAction::UP);

  std::cout << puzzle.getState() << std::endl;

  puzzle.move(STPAction::DOWN);
  puzzle.move(STPAction::DOWN);

  std::cout << puzzle.getState() << std::endl;

  puzzle.move(STPAction::LEFT);
  puzzle.move(STPAction::LEFT);

  std::cout << puzzle.getState() << std::endl;

  puzzle.move(STPAction::UP);
  puzzle.move(STPAction::UP);

  std::cout << puzzle.getState() << std::endl;
}
