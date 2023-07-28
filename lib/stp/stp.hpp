#ifndef PDB_ANN_STP
#define PDB_ANN_STP

#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <tuple>
#include <optional>

enum STPAction
{
  UP,
  RIGHT,
  DOWN,
  LEFT,
  NO
};

class STP
{
private:
  int _height;
  int _width;
  int _blank;
  torch::Tensor _state;

public:
  STP(int width, int height);
  int size();
  int blank();
  void initGoal();
  void initState(torch::Tensor state);
  void initState(std::vector<int> state);
  torch::Tensor getState();
  void toAbstract(std::vector<int> pattern);
  torch::Tensor getFlattenState(bool by_pos = false);
  torch::Tensor getFlattenState(std::vector<int> pattern);
  int hashState(std::optional<torch::Tensor> pi_optional = std::nullopt);
  int hashState(std::vector<int> pattern);
  bool move(STPAction action);
  std::optional<std::tuple<torch::Tensor, int>> nextState(STPAction action, int tile);
  std::vector<STPAction> getActions(int tile);
  std::vector<torch::Tensor> getSuccessors(int tile);
};

#endif