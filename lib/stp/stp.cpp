#include "stp.hpp"

STP::STP(unsigned int width, unsigned int height)
{
  _width = width;
  _height = height;

  auto options = torch::TensorOptions().dtype(torch::kInt);

  _state = torch::zeros({_height, _width}, options);
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
}

void STP::initState(int *state)
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

  auto options = torch::TensorOptions().dtype(torch::kInt);
  torch::Tensor pi_1 = torch::zeros({pi_size}, options);

  for (int i = 0; i < pi_size; i++)
  {
    pi_1[pi[i]] = i;
  }

  torch::Scalar n = torch::Scalar(pi_size);

  return rank(n, pi, pi_1);
}

int STP::hashState(std::vector<int> pattern)
{
  torch::Tensor pi = getFlattenState(pattern);
  return hashState(pi);
}

int STP::rank(torch::Scalar n, torch::Tensor pi, torch::Tensor pi_1)
{
  if (n.equal(1))
    return 0;

  int s = pi[n.toInt() - 1].item<int>();

  int a = pi[n.toInt() - 1].item<int>();
  int b = pi[pi_1[n.toInt() - 1]].item<int>();

  pi[n.toInt() - 1] = b;
  pi[pi_1[n.toInt() - 1]] = a;

  a = pi_1[s].item<int>();
  b = pi_1[n.toInt() - 1].item<int>();

  pi_1[s] = b;
  pi_1[n.toInt() - 1] = a;

  return s + n.toInt() * rank(n.toInt() - 1, pi, pi_1);
}
