#include <torch/torch.h>
#include <iostream>
#include <vector>

#include "stp/stp.hpp"
#include "pdb/pdb.hpp"

int main()
{
  STP puzzle = STP(5, 5);
  puzzle.initGoal();

  std::vector<int> pattern = std::vector<int>({1, 2, 3, 4, 5, 6, 7});

  PDB pdb = PDB(puzzle, pattern);

  std::cout << pdb.size() << std::endl;

  pdb.fill();

  std::cout << pdb.getTable() << std::endl;
}
