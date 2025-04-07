# End-to-end ML algorithms and implementations

This repository contains base (no packages used) machine learning and AI algorithms implemented in Jupyter notebooks.

- **BaseML_algos.ipynb**  
  This notebook demonstrates foundational machine learning algorithms, including linear models, support vector machines, decision trees, and clustering techniques. It provides comprehensive examples of data preprocessing, model training, hyperparameter tuning, and evaluation metrics, serving as a solid base for understanding classical ML methods.

- **Bayesian_networks.ipynb**  
  This notebook implements Bayesian network models using probabilistic graphical modeling techniques. It covers the construction of network structures, parameter estimation via conditional probability tables, and inference, offering insights into reasoning under uncertainty.

- **Neural Network Models**
  - **Convolutional Neural Networks (CNNs):** Implementation of CNNs for image processing tasks. The code provides an end-to-end pipeline for feature extraction and classification using convolutional layers.
  - **Multi-Layer Perceptrons (MLPs):** Fully-connected neural network models for general-purpose classification and regression tasks. The code is structured to allow easy experimentation with different architectures.

- **Path Planning**
  - **Route Finding Algorithm:** A search-based algorithm designed for navigating through complex terrains. The implementation takes into account dynamic factors such as footwear suitability on various terrains (rock, ice, grass) and includes mechanisms for state augmentation (position and current equipment). This module demonstrates the application of graph search techniques (Dijkstra/A*) in a simulated exploration scenario.

- **Propositional Logic Implementations**
  - **Wumpus World:** An implementation of the Wumpus world problem that utilizes propositional logic to determine safe moves in a hazardous environment. The code is organized to illustrate knowledge representation and inference.
  - **Three Doors Puzzle:** A logic-based simulation that explores decision-making under uncertainty using propositional reasoning. The implementation is a concise demonstration of logical inference and problem solving.

- **Random Forest Classifier.ipynb**  
  An end-to-end example of training and evaluating a Random Forest classifier. It demonstrates data preprocessing, model training with hyperparameter tuning, and evaluation using performance metrics such as accuracy and precision, focusing on classification tasks with scikit-learn.

- **Reinforcement learning.ipynb**  
  RL implementations- includes algorithms such as Q-learning or policy gradient methods, illustrating how an agent interacts with an environment to learn optimal policies through exploration and exploitation, and it tracks performance improvements over episodes.

### Prerequisites

Ensure that you have Python (3.8 to 3.11 recommended) installed along with the following packages:
- NumPy
- scikit-learn
- TensorFlow / Keras (for neural network models)
- Matplotlib, Seaborn
- Other dependencies as specified in the individual notebooks

### Contributing
Contributions are welcome. Please fork the repository and create a pull request for any enhancements or bug fixes. All contributions must adhere to the coding standards outlined in the repository.

### License
This project is licensed under the MIT License. See the LICENSE file for details.

### Final Remarks
This repository is intended to be both a learning tool and a reference for implementing diverse AI algorithms. The code is structured to emphasize clarity and modularity, ensuring that each module can be easily extended or integrated into larger projects. If you encounter any issues or have suggestions for improvement, please open an issue or submit a pull request.
