# ResNet18 on CIFAR-100

This project implements the ResNet18 model to classify images from the CIFAR-100 dataset. It includes scripts for training and testing the model, as well as utilities for data preprocessing and model management.

## Project Structure

```
resnet18-cifar100
├── src
│   ├── model.py        # Contains the ResNet18 model definition
│   ├── train.py        # Script for training the model
│   ├── test.py         # Script for testing the model
│   └── utils.py        # Utility functions for data handling and model management
├── data
│   └── README.md       # store the downloaded CIFAR-100 dataset on first run
├── requirements.txt    # List of project dependencies
├── .gitignore          # Files and directories to ignore in Git
└── README.md           # Project documentation
```

## Setup

1. Clone the repository:
   ```
   git clone <repository-url>
   cd resnet18-cifar100
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Training the Model

To train the ResNet18 model on the CIFAR-100 dataset, run the following command:

```
python src/train.py
```

## Testing the Model

After training, you can evaluate the model's performance using:

```
python src/test.py
```

## Dataset Information

The CIFAR-100 dataset can be downloaded from [here](https://www.cs.toronto.edu/~kriz/cifar.html). Follow the instructions in the `data/README.md` file for details on how to prepare the dataset for training and testing.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.