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
  int x_b = static_cast<int>(_blank / _width);
  int y_b = _blank % _width;

  switch (action)
  {
  case UP:
    if (x_b > 0)
    {
      int tmp = _state[x_b - 1][y_b].item<int>();
      _state[x_b - 1][y_b] = 0;
      _state[x_b][y_b] = tmp;
      _blank = (x_b - 1) * _width + y_b;
    }
    else
      return false;
    break;
  case RIGHT:
    if (y_b < _width - 1)
    {
      int tmp = _state[x_b][y_b + 1].item<int>();
      _state[x_b][y_b + 1] = 0;
      _state[x_b][y_b] = tmp;
      _blank = (x_b)*_width + (y_b + 1);
    }
    else
      return false;
    break;
  case DOWN:
    if (x_b < _height - 1)
    {
      int tmp = _state[x_b + 1][y_b].item<int>();
      _state[x_b + 1][y_b] = 0;
      _state[x_b][y_b] = tmp;
      _blank = (x_b + 1) * _width + y_b;
    }
    else
      return false;
    break;
  case LEFT:
    if (y_b > 0)
    {
      int tmp = _state[x_b][y_b - 1].item<int>();
      _state[x_b][y_b - 1] = 0;
      _state[x_b][y_b] = tmp;
      _blank = (x_b)*_width + (y_b - 1);
    }
    else
      return false;
    break;

  default:
    return false;
  }

  return true;
}