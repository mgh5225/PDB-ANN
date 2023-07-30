#include <torch/torch.h>
#include <iostream>
#include <vector>

#include "stp/stp.hpp"
#include "pdb/pdb.hpp"
#include "dataset/dataset.hpp"

int main()
{
  STPDataset::generateRandom("data/dataset.json");
}
