#include "stp.hpp"

STP::STP(int width, int height)
{
  _width = width;
  _height = height;

  auto options = torch::TensorOptions().dtype(torch::kInt);

  _state = torch::zeros({_height, _width}, options);
  _blank = 0;
}

int STP::size()
{
  return _width * _height;
}

int STP::blank()
{
  return _blank;
}

void STP::initGoal()
{
  int i = 0;
  int j = 0;
  for (int n = 0; n < size(); n++)
  {
    _state[i][j] = n;
    j++;
    j %= _width;
    if (j == 0)
      i++;
  }

  _blank = 0;
}

void STP::initState(std::vector<int> state)
{
  int i = 0;
  int j = 0;
  for (int n = 0; n < size(); n++)
  {
    _state[i][j] = state[n];
    j++;
    j %= _width;
    if (j == 0)
      i++;

    if (state[n] == 0)
      _blank = n;
  }
}

void STP::initState(torch::Tensor state)
{
  _state.copy_(state);
}

torch::Tensor STP::getState()
{
  return _state;
}

void STP::toAbstract(std::vector<int> pattern)
{
  std::vector<int> state = std::vector<int>(size(), -1);
  torch::Tensor f_state = getFlattenState(true);

  for (int i = 0; i < pattern.size(); i++)
  {
    state[f_state[pattern[i]].item<int>()] = pattern[i];
  }

  initState(state);
}

torch::Tensor STP::getFlattenState(bool by_pos)
{
  torch::Tensor f_state = _state.flatten();

  if (by_pos)
  {
    auto options = torch::TensorOptions().dtype(torch::kInt);
    torch::Tensor pos = torch::zeros({size()}, options);

    for (int i = 0; i < size(); i++)
    {
      pos[f_state[i]] = i;
    }

    return pos;
  }
  return f_state;
}

torch::Tensor STP::getFlattenState(std::vector<int> pattern)
{
  torch::Tensor f_state = getFlattenState(true);

  auto options = torch::TensorOptions().dtype(torch::kInt);
  torch::Tensor res = torch::zeros({(int)pattern.size()}, options);

  for (int i = 0; i < pattern.size(); i++)
  {
    res[i] = f_state[pattern[i]];
  }

  return res;
}

int STP::hashState(std::optional<torch::Tensor> pi_optional)
{
  torch::Tensor pi = pi_optional.value_or(getFlattenState());

  int pi_size = pi.size(0);

  int k = static_cast<int>(std::log2f(pi_size));

  auto options = torch::TensorOptions().dtype(torch::kInt);
  torch::Tensor pi_1 = torch::zeros({((1 << (1 + k)) - 1)}, options);

  int rank = 0;

  for (int i = 0; i < pi_size; i++)
  {
    int counter = pi[i].item<int>();
    int node = (1 << k) - 1 + counter;

    for (int j = 0; j < k; j++)
    {
      int isEven = (1 - (node & 1));
      counter -= isEven * (pi_1[(node - 1) >> 1].item<int>() - pi_1[node].item<int>());
      pi_1[node] += 1;
      node = (node - 1) >> 1;
    }
    pi_1[node] += 1;
    rank = rank * (size() - i) + counter;
  }

  return rank;
}

int STP::hashState(std::vector<int> pattern)
{
  torch::Tensor pi = getFlattenState(pattern);

  return hashState(pi);
}

bool STP::move(STPAction action)
{
  auto res = nextState(action, _blank);
  if (res.has_value())
  {
    initState(std::get<torch::Tensor>(res.value()));
    _blank = std::get<int>(res.value());
  }
  return false;
}

std::optional<std::tuple<torch::Tensor, int>> STP::nextState(STPAction action, int tile)
{
  int x_t = static_cast<int>(tile / _width);
  int y_t = tile % _width;

  torch::Tensor n_state = 1 * getState();

  switch (action)
  {
  case UP:
    if (x_t > 0)
    {
      int tmp = n_state[x_t - 1][y_t].item<int>();
      n_state[x_t - 1][y_t] = 0;
      n_state[x_t][y_t] = tmp;
      tile = (x_t - 1) * _width + y_t;
    }
    else
      return std::nullopt;
    break;
  case RIGHT:
    if (y_t < _width - 1)
    {
      int tmp = n_state[x_t][y_t + 1].item<int>();
      n_state[x_t][y_t + 1] = 0;
      n_state[x_t][y_t] = tmp;
      tile = (x_t)*_width + (y_t + 1);
    }
    else
      return std::nullopt;
    break;
  case DOWN:
    if (x_t < _height - 1)
    {
      int tmp = n_state[x_t + 1][y_t].item<int>();
      n_state[x_t + 1][y_t] = 0;
      n_state[x_t][y_t] = tmp;
      tile = (x_t + 1) * _width + y_t;
    }
    else
      return std::nullopt;
    break;
  case LEFT:
    if (y_t > 0)
    {
      int tmp = n_state[x_t][y_t - 1].item<int>();
      n_state[x_t][y_t - 1] = 0;
      n_state[x_t][y_t] = tmp;
      tile = (x_t)*_width + (y_t - 1);
    }
    else
      return std::nullopt;
    break;

  default:
    return std::nullopt;
  }

  return std::tuple<torch::Tensor, int>({n_state, tile});
}

std::vector<STPAction> STP::getActions(int tile)
{
  std::vector<STPAction> actions = std::vector<STPAction>();

  int x_t = static_cast<int>(tile / _width);
  int y_t = tile % _width;

  if (x_t > 0)
    actions.push_back(STPAction::UP);
  if (y_t < _width - 1)
    actions.push_back(STPAction::RIGHT);
  if (x_t < _height - 1)
    actions.push_back(STPAction::DOWN);
  if (y_t > 0)
    actions.push_back(STPAction::LEFT);

  return actions;
}

std::vector<torch::Tensor> STP::getSuccessors(int tile)
{
  std::vector<STPAction> actions = getActions(tile);
  std::vector<torch::Tensor> successors = std::vector<torch::Tensor>();

  for (auto &action : actions)
  {
    auto n_state = nextState(action, tile);
    if (n_state.has_value())
    {
      successors.push_back(std::get<torch::Tensor>(n_state.value()));
    }
  }

  return successors;
}