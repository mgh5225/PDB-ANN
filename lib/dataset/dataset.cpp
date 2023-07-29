#include "dataset.hpp"

STPDataset::STPDataset(PDBs pdb_s)
{
  _pdb_s = pdb_s;

  for (auto &pdb : _pdb_s)
  {
    pdb.fill();
  }
}

STPDataset::STPDataset(STP goalSTP, std::vector<std::vector<int>> pattern_s)
{
  _pdb_s = PDBs();

  for (auto &pattern : pattern_s)
  {
    _pdb_s.push_back(PDB(goalSTP, pattern));
  }

  for (auto &pdb : _pdb_s)
  {
    pdb.fill();
  }
}

void STPDataset::generateRandom(int size)
{
}
