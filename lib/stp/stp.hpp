#ifndef PDB_ANN_STP
#define PDB_ANN_STP

#include <torch/torch.h>
#include <iostream>

class STP
{
private:
  unsigned int _height;
  unsigned int _width;
  torch::Tensor _state;

public:
  STP(unsigned int width, unsigned int height);
  void initGoal();
  void initState(torch::Tensor state);
  torch::Tensor getState();
  unsigned int hashState();
};

#endif