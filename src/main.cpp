#include <torch/torch.h>
#include <iostream>
#include <vector>

#include "stp/stp.hpp"
#include "pdb/pdb.hpp"

int main()
{
  STP puzzle = STP(3, 3);
  puzzle.initGoal();

  std::vector<int> pattern = std::vector<int>({1, 2});

  PDB pdb = PDB(puzzle, pattern);

  std::cout << pdb.size() << std::endl;

  pdb.fill();

  std::cout << pdb.getTable() << std::endl;
}
