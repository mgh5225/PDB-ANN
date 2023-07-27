#ifndef PDB_ANN_STP
#define PDB_ANN_STP

#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <optional>

class STP
{
private:
  unsigned int _height;
  unsigned int _width;
  torch::Tensor _state;

public:
  STP(unsigned int width, unsigned int height);
  int size();
  void initGoal();
  void initState(torch::Tensor state);
  void initState(int *state);
  torch::Tensor getState();
  torch::Tensor getFlattenState(bool by_pos = false);
  torch::Tensor getFlattenState(std::vector<int> pattern);
  int hashState(std::optional<torch::Tensor> pi_optional = std::nullopt);
  int hashState(std::vector<int> pattern);
};

#endif