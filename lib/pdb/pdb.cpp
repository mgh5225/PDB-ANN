#include "pdb.hpp"

PDB::PDB(STP goalSTP, std::vector<int> pattern)
{
  int pdbSize = 1;
  for (int i = goalSTP.size(); i > goalSTP.size() - pattern.size(); i--)
  {
    pdbSize *= i;
  }

  auto options = torch::TensorOptions().dtype(torch::kInt);
  _table = torch::zeros({pdbSize}, options);

  goalSTP.toAbstract(pattern);
  _pattern = goalSTP.getState();
}

int PDB::size()
{
  return _table.size(0);
}

void PDB::fill()
{
}
