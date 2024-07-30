# Physics-informed-Machine-Learning-for-Autonomous-Vehicle-Control

## Project Overview
This project aims to explore and implement control strategies for autonomous vehicles using physics-informed machine learning methods. The approach combines physical models with deep learning techniques to predict and optimize vehicle control strategies for efficient and safe autonomous driving.

## Usage Instructions

1. **Generate Training Data**
   First, run `DataSetGeneration.py` to generate the training data `Train Data.csv` and test data `Test Data.csv`.
   ```bash
   python DataSetGeneration.py
   ```

2. **Train Neural Network Controller**
   Next, run `DPC_train_GPU.py` to train the neural network controller.
   ```bash
   python DPC_train_GPU.py
   ```

3. **Simulation**
   - Perform simulations for different scenarios:
     - Cruising Control: Run `mainCruising.py`
       ```bash
       python mainCruising.py
       ```
     - Lane Changing Control: Run `mainLaneChanging.py`
       ```bash
       python mainLaneChanging.py
       ```
     - Generalization Test: Run `mainGeneralization.py`
       ```bash
       python mainGeneralization.py
       ```

## File Descriptions
- `DataSetGeneration.py`: Generates training and test data.
- `DPC_train_GPU.py`: Trains the neural network controller.
- `mainCruising.py`: Simulates cruising control.
- `mainLaneChanging.py`: Simulates lane changing control.
- `mainGeneralization.py`: Performs generalization tests.

## Dependencies
The project requires the following libraries:
- numpy
- pandas
- matplotlib
- torch
- casadi

Please ensure you have these dependencies installed before running the code. You can install the required Python libraries using the following command:
```bash
pip install numpy pandas matplotlib torch casadi
```

## Author
Xianning Li  
New York University  
Email: xl5305@nyu.edu
