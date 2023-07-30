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
  STP();
  STP(int width, int height);
  STP(std::tuple<int, int> dimension);
  STP(const STP &_stp);
  int size();
  int blank();
  int getTile(int tile);
  std::tuple<int, int> dimension();
  void initGoal();
  void initState(torch::Tensor state);
  void initState(std::vector<int> state);
  torch::Tensor getState();
  void toAbstract(std::vector<int> pattern);
  torch::Tensor getFlattenState(bool by_pos = false);
  torch::Tensor getFlattenState(std::vector<int> pattern);
  int64_t hashState(std::optional<torch::Tensor> pi_optional = std::nullopt);
  int64_t hashState(std::vector<int> pattern, std::optional<std::vector<int>> pi_helper = std::nullopt);
  bool moveBlank(STPAction action);
  std::optional<std::tuple<torch::Tensor, int>> nextState(STPAction action, int tile);
  std::vector<STPAction> getActions(int tile);
  std::vector<std::tuple<STP, int>> getSuccessors(int tile);
  int getMDHeuristic(std::optional<torch::Tensor> state_optional = std::nullopt);
};

#endif