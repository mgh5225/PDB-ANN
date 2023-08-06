#include "model.hpp"

QNT::QNT()
{
  conv = register_module("conv", torch::nn::Conv2d(4, 32, 2));
  fc1 = register_module("fc1", torch::nn::Linear(128, 256));
  fc2 = register_module("fc2", torch::nn::Linear(256, 512));
  fc3 = register_module("fc3", torch::nn::Linear(512, 164));
  fc4 = register_module("fc4", torch::nn::Linear(164, 6));
}

torch::Tensor QNT::forward(torch::Tensor x)
{
  x = torch::relu(conv->forward(x));
  x = torch::flatten(x, 1);
  x = torch::relu(fc1->forward(x));
  x = torch::relu(fc2->forward(x));
  x = torch::relu(fc3->forward(x));
  x = torch::softmax(fc4->forward(x), 1);

  return x;
}

int QNT::getHeuristic(torch::Tensor v, float q)
{
  torch::NoGradGuard no_grad;

  torch::Tensor p_hc = forward(v);
  float c_h = 0;
  int i = 0;

  for (; i < p_hc.size(1); i++)
  {
    float p_hc_i = p_hc[0][i].item<float>();
    c_h += p_hc_i;
    if (c_h >= q)
      break;
  }

  return i;
}

torch::Tensor QNT::getHeuristic(torch::Tensor v, torch::Tensor q)
{
  torch::NoGradGuard no_grad;

  torch::Tensor p_hc = forward(v);

  int64_t num_classes = p_hc.size(1);

  torch::Tensor c_h = torch::cumsum(p_hc, 1);
  torch::Tensor h = torch::ge(c_h, q.reshape({-1, 1}));

  return num_classes - torch::sum(h, 1);
}

void QNT::train(json params)
{
  json adam_params = params["adam"];
  json step_lr_params = params["step_lr"];

  double alpha = adam_params["alpha"];
  std::tuple<double, double> betas = adam_params["beta"];
  double epsilon = adam_params["epsilon"];

  double gamma = step_lr_params["gamma"];
  unsigned int step_size = step_lr_params["step_size"];

  int64_t batch_size = params["batch_size"];
  json j_dataset = params["dataset"];
  std::string dataset_path = j_dataset["path"];
  int64_t epochs = params["epochs"];
  double random_split = params["random_split"];
  int heuristic_idx = params["heuristic_idx"];

  auto adamOptions = torch::optim::AdamOptions().lr(alpha).betas(betas).eps(epsilon);

  auto cross_entropy = torch::nn::CrossEntropyLoss();
  auto adam = torch::optim::Adam(parameters(), adamOptions);
  auto stepLR = torch::optim::StepLR(adam, step_size, gamma);

  auto dataset = STPDataset(dataset_path, random_split);

  std::tuple<STPDataset::STPSubset, STPDataset::STPSubset> datasets = dataset.splitDataset(heuristic_idx);

  auto dataLoaderOptions = torch::data::DataLoaderOptions().batch_size(batch_size);

  auto train_dataset = std::get<0>(datasets).map(torch::data::transforms::Stack<>());
  auto test_dataset = std::get<1>(datasets).map(torch::data::transforms::Stack<>());

  auto train_data_loader = torch::data::make_data_loader(std::move(train_dataset), dataLoaderOptions);
  auto test_data_loader = torch::data::make_data_loader(std::move(test_dataset), dataLoaderOptions);

  std::cout << "Epoch"
            << "\t"
            << "Mean Loss" << std::endl;

  for (int64_t epoch = 0; epoch < epochs; epoch++)
  {
    for (auto &batch : *train_data_loader)
    {
      torch::Tensor data = batch.data;
      torch::Tensor target = batch.target;

      adam.zero_grad();

      torch::Tensor classes = forward(data);

      torch::Tensor loss = cross_entropy->forward(classes, target);

      loss.backward();
      adam.step();
    }

    {
      torch::NoGradGuard no_grad;

      torch::Tensor sum_loss = torch::zeros({1});

      for (auto &batch : *test_data_loader)
      {
        torch::Tensor data = batch.data;
        torch::Tensor target = batch.target;

        torch::Tensor classes = forward(data);

        torch::Tensor loss = cross_entropy->forward(classes, target);

        sum_loss += loss;
      }

      std::cout << std::fixed << std::setfill('0') << std::setw(5) << epoch << "\t" << std::setw(10) << torch::mean(sum_loss).item<double>() << std::endl;
    }

    stepLR.step();
  }
}

void QNT::saveQNT()
{
  auto qnt = std::make_shared<QNT>(*this);
  torch::save(qnt, "data/qnt.pt");
}

std::shared_ptr<QNT> QNT::loadQNT()
{
  auto qnt = std::make_shared<QNT>();
  torch::load(qnt, "data/qnt.pt");
  return qnt;
}

double QNT::findQStar(json params)
{
  torch::NoGradGuard no_grad;

  int64_t batch_size = params["batch_size"];
  json j_dataset = params["dataset"];
  int heuristic_idx = params["heuristic_idx"];

  std::string dataset_path = j_dataset["path"];

  auto dataset = STPDataset(dataset_path, 1);

  std::tuple<STPDataset::STPSubset, STPDataset::STPSubset> datasets = dataset.splitDataset(heuristic_idx);

  auto dataLoaderOptions = torch::data::DataLoaderOptions().batch_size(batch_size);

  auto whole_dataset = std::get<0>(datasets).map(torch::data::transforms::Stack<>());

  auto whole_data_loader = torch::data::make_data_loader(std::move(whole_dataset), dataLoaderOptions);

  double q_star = 1;

  for (auto &batch : *whole_data_loader)
  {
    torch::Tensor data = batch.data;
    torch::Tensor target = batch.target;

    torch::Tensor f_target = target.flip(1);
    torch::Tensor s_target = std::get<0>(f_target.cummax(1)).flip(1);

    torch::Tensor classes = forward(data);

    torch::Tensor q_v = torch::sum(torch::mul(s_target, classes), 1);

    double min_q_v = torch::min(q_v).item<double>();

    if (min_q_v < q_star)
      q_star = min_q_v;
  }

  return q_star;
}

std::vector<std::tuple<std::string, int, int>> QNT::run(json params)
{
  torch::NoGradGuard no_grad;

  auto res = std::vector<std::tuple<std::string, int, int>>();

  int64_t batch_size = params["batch_size"];
  json j_dataset = params["dataset"];
  double q_star = params["q_star"];
  int heuristic_idx = params["heuristic_idx"];

  std::string dataset_path = j_dataset["path"];

  auto dataset = STPDataset(dataset_path, 1);

  std::tuple<STPDataset::STPSubset, STPDataset::STPSubset> datasets = dataset.splitDataset(heuristic_idx);

  auto dataLoaderOptions = torch::data::DataLoaderOptions().batch_size(batch_size);

  STPDataset::STPSubset whole_dataset = std::get<0>(datasets);
  auto map_whole_dataset = whole_dataset.map(torch::data::transforms::Stack<>());

  auto whole_data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(map_whole_dataset), dataLoaderOptions);

  int64_t batch_idx = 0;

  for (auto &batch : *whole_data_loader)
  {
    torch::Tensor data = batch.data;
    torch::Tensor target = batch.target;

    torch::Tensor f_target = target.flip(1);
    torch::Tensor s_target = std::get<0>(f_target.cummax(1)).flip(1);
    torch::Tensor h_target = torch::sum(s_target, 1) - 1;

    torch::Tensor q = torch::full({data.size(0)}, q_star);

    torch::Tensor h = getHeuristic(data, q);

    for (int64_t i = 0; i < data.size(0); i++)
    {
      json row = whole_dataset.at(batch_idx + i);
      std::vector<int> permutation = row["permutation"];

      std::stringstream state;

      std::copy(permutation.begin(), permutation.end(), std::ostream_iterator<int>(state, " "));

      res.push_back(std::make_tuple(state.str(), h_target[i].item<int>(), h[i].item<int>()));
    }

    batch_idx += data.size(0);
  }

  return res;
}
