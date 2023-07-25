# PDB-ANN

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Overview

The PDB-ANN project aims to implement one of the methodologies mentioned in the research paper - [Optimal Search with Neural Networks: Challenges and Approaches](https://webdocs.cs.ualberta.ca/~nathanst/papers/li2022optimalnn.pdf).

## Requirements

Before you begin, ensure you have met the following requirements:

- You have installed the latest version of [Libtorch](https://pytorch.org/cppdocs/installing.html). Detailed instructions related to the installation of Libtorch can be found on the linked page.

## Installation

To build PDB-ANN, follow these steps:

Linux:

```bash
git clone https://github.com/mgh5225/PDB-ANN.git
cd PDB-ANN
bash build.sh
```

Please be aware that you'll need to update the absolute path to Libtorch in the `build.sh` script.

## Usage

Once the build process is complete, you can initiate the program using the following commands:

```bash
cd build
./main
```

## License

This project uses the following license: [MIT license](https://github.com/mgh5225/PDB-ANN/blob/main/LICENSE)
