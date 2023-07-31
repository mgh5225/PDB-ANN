#include "model.hpp"

QNT::QNT()
{
  conv = register_module("conv", torch::nn::Conv2d(4, 32, 3));
  fc1 = register_module("fc1", torch::nn::Linear(32, 128));
  fc2 = register_module("fc2", torch::nn::Linear(128, 256));
  fc3 = register_module("fc3", torch::nn::Linear(256, 6));
}

torch::Tensor QNT::forward(torch::Tensor x)
{
  x = torch::relu(conv->forward(x));
  x = torch::relu(fc1->forward(torch::flatten(x)));
  x = torch::relu(fc2->forward(x));
  x = torch::softmax(fc3->forward(x), 1);

  return x;
}

int QNT::getHeuristic(torch::Tensor v, float q)
{
  torch::Tensor p_hc = forward(v);
  int c_h = 0;
  int i = 0;

  for (; i < p_hc.size(0); i++)
  {
    float p_hc_i = p_hc[i].item<float>();
    c_h += p_hc_i;
    if (c_h >= q)
      break;
  }

  return i;
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

  auto adamOptions = torch::optim::AdamOptions().lr(alpha).betas(betas).eps(epsilon);

  auto cross_entropy = torch::nn::CrossEntropyLoss();
  auto adam = torch::optim::Adam(parameters(), adamOptions);
  auto stepLR = torch::optim::StepLR(adam, step_size, gamma);

  auto dataset = STPDataset(dataset_path, random_split);

  std::tuple<STPDataset::STPSubset, STPDataset::STPSubset> datasets = dataset.splitDataset();

  auto dataLoaderOptions = torch::data::DataLoaderOptions().batch_size(batch_size);

  auto train_dataset = std::get<0>(datasets).map(torch::data::transforms::Stack<>());
  auto test_dataset = std::get<1>(datasets).map(torch::data::transforms::Stack<>());

  auto train_data_loader = torch::data::make_data_loader(std::move(train_dataset), dataLoaderOptions);
  auto test_data_loader = torch::data::make_data_loader(std::move(test_dataset), dataLoaderOptions);

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

      for (auto &batch : *train_data_loader)
      {
        torch::Tensor data = batch.data;
        torch::Tensor target = batch.target;

        torch::Tensor classes = forward(data);

        torch::Tensor loss = cross_entropy->forward(classes, target);

        std::cout << torch::mean(loss) << std::endl;
      }
    }

    stepLR.step();
  }
}
