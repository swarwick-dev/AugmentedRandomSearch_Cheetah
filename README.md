<p align="center">
  <img src="gymtrainer-ai.png" width="60%" alt="GYMTRAINER-AI-logo">
</p>
<p align="center">
    <h1 align="center">GYMTRAINER-AI</h1>
</p>
<p align="center">
    <em>Welcome to **GYMTRAINER-AI**, a Python application that allows users to select and train reinforcement learning models in various gymnasium environments. This project aims to provide an easy-to-use interface for experimenting with different RL algorithms and environments, making it an excellent tool for both beginners and experienced practitioners in the field of reinforcement learning.</em>
</p>
<p align="center">
	<img src="https://img.shields.io/github/license/swarwick-dev/gymtrainer-ai?style=plastic&logo=opensourceinitiative&logoColor=white&color=0080ff" alt="license">
	<img src="https://img.shields.io/github/last-commit/swarwick-dev/gymtrainer-ai?style=plastic&logo=git&logoColor=white&color=0080ff" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/swarwick-dev/gymtrainer-ai?style=plastic&color=0080ff" alt="repo-top-language">
	<img src="https://img.shields.io/github/languages/count/swarwick-dev/gymtrainer-ai?style=plastic&color=0080ff" alt="repo-language-count">
</p>
<p align="center">
		<em>Built with the tools and technologies:</em>
</p>
<p align="center">
	<img src="https://img.shields.io/badge/Python-3776AB.svg?style=plastic&logo=Python&logoColor=white" alt="Python">
	<img src="https://img.shields.io/badge/NumPy-013243.svg?style=plastic&logo=NumPy&logoColor=white" alt="NumPy">
</p>

<br>

#####  Table of Contents

- [ Overview](#-overview)
- [ Features](#-features)
- [ Repository Structure](#-repository-structure)
- [ Modules](#-modules)
- [ Getting Started](#-getting-started)
    - [ Prerequisites](#-prerequisites)
    - [ Installation](#-installation)
    - [ Usage](#-usage)
    - [ Tests](#-tests)
- [ Project Roadmap](#-project-roadmap)
- [ Contributing](#-contributing)
- [ License](#-license)
- [ Acknowledgments](#-acknowledgments)

---

##  Overview

**GYMTRAINER-AI** is a Python application designed to facilitate the training of reinforcement learning (RL) models in various gymnasium environments. The application provides a user-friendly interface built with PySide6, allowing users to easily select and configure different RL models and environments.

### Key Features

- **User Interface**: The application features a graphical user interface (GUI) built with PySide6, making it accessible for users to interact with the application without needing to write code.
- **Reinforcement Learning Techniques**: Currently, the application supports several RL techniques, with a focus on Augmented Random Search (ARS). ARS is a simple yet effective method for training RL models, particularly in high-dimensional continuous control tasks.
- **Training Visualization**: Training progress can be monitored using TensorBoard, which provides real-time visualizations of various training metrics. This helps users understand how their models are performing and make necessary adjustments.
- **Video Storage**: The application periodically stores videos of the models' training sessions. These videos allow users to visually inspect the progress and behavior of their models over time.

### Reinforcement Learning Techniques

- **Augmented Random Search (ARS)**: ARS is a derivative-free optimization algorithm that is particularly useful for training policies in high-dimensional spaces. It works by perturbing the policy parameters with random noise and updating the parameters based on the performance of these perturbations.
- **Other Techniques**: The application is designed to be extensible, allowing for the integration of additional RL algorithms in the future.

By combining these features, **GYMTRAINER-AI** provides a comprehensive platform for experimenting with and optimizing reinforcement learning models in a variety of environments.

---

##  Features

|    |   Feature         | Description |
|----|-------------------|---------------------------------------------------------------|
| ⚙️  | **Architecture**  | The project's architecture is modular, with separate modules for data processing (Normaliser), AI model training (Trainer), and file management (Helpers). It utilizes a BaseModel as the foundation for policy weights and hyperparameters. |
| 🔌 | **Integrations**  | The project integrates with various libraries and frameworks, including NumPy, Gymnasium, PyTorch, TensorBoard, and PySide6. It also uses external dependencies like requirements.txt and setup.py. |
| 🧩 | **Modularity**    | The codebase is designed with modularity in mind, allowing for easy reuse of individual modules or classes. This facilitates experimentation and extension of the project's capabilities. |
| 📦 | **Dependencies**  | The project depends on several external libraries and dependencies, including requirements.txt, setup.py, NumPy, Gymnasium, PyTorch, TensorBoard, PySide6, moviepy, and text. |

---

##  Repository Structure

```sh
└── gymtrainer-ai/
    ├── LICENSE
    ├── README.md
    ├── __pycache__
    │   └── main.cpython-311.pyc
    ├── gui
    │   ├── __init__.py
    │   ├── __pycache__
    │   └── main_window.py
    ├── models
    │   ├── __init__.py
    │   ├── __pycache__
    │   ├── ars_model.py
    │   ├── base_model.py
    │   └── hp.py
    ├── requirements.txt
    ├── setup.py
    └── utils
        ├── __init__.py
        ├── __pycache__
        ├── helpers.py
        ├── normaliser.py
        └── trainer.py
```

---

##  Modules

<details closed><summary>.</summary>

| File | Summary |
| --- | --- |
| [requirements.txt](https://github.com/swarwick-dev/gymtrainer-ai/blob/main/requirements.txt) | This requirements file is critical in structuring dependencies for the GymTrainer-ai repository. It specifies essential packages like NumPy, Gymnasium, and PyTorch to develop intelligent exercise training software with visualization capabilities through TensorBoard and user-friendly interfaces using PySide6. |
| [setup.py](https://github.com/swarwick-dev/gymtrainer-ai/blob/main/setup.py) | Orchestrate AI Training ConfigurationsThis setup script enables installation and configuration of AI training modules within the GymTrainer-AI repository. It specifies dependencies, packages, and entry points for console scripts, facilitating seamless integration with various libraries and frameworks. |

</details>

<details closed><summary>utils</summary>

| File | Summary |
| --- | --- |
| [normaliser.py](https://github.com/swarwick-dev/gymtrainer-ai/blob/main/utils/normaliser.py) | Standardising is a crucial component in GymTrainer-AIs architecture, enabling effective data processing. Normaliser calculates statistical parameters like mean and variance for given inputs, allowing for observation-based adjustments and precise normalisation. |
| [trainer.py](https://github.com/swarwick-dev/gymtrainer-ai/blob/main/utils/trainer.py) | Trainer class enables efficient exploration and training of various AI models within the Gym-Trainer-AI repository. It coordinates interactions with the environment, manages model updates, and logs progress using TensorBoard. The Trainer facilitates adaptive exploration strategies and early stopping, allowing for optimal policy learning. |
| [helpers.py](https://github.com/swarwick-dev/gymtrainer-ai/blob/main/utils/helpers.py) | Organizing. The helpers file in the utils module enables directory creation and checks for the existence of base directories within the gymtrainer-ai repositorys structure. This crucial feature facilitates efficient management of files and folders, ensuring that the model training process runs smoothly. |

</details>

<details closed><summary>models</summary>

| File | Summary |
| --- | --- |
| [base_model.py](https://github.com/swarwick-dev/gymtrainer-ai/blob/main/models/base_model.py) | A foundational component in the gym trainer AI repository. The BaseModel defines the fundamental structure for policy weights and hyperparameters. It provides methods for evaluation, sampling, updating, saving, and loading model states, enabling seamless integration with other models in the repository. |
| [ars_model.py](https://github.com/swarwick-dev/gymtrainer-ai/blob/main/models/ars_model.py) | Optimizing Neural Networks Architecture ARSModel script refines neural network architecture by introducing directional search for optimal parameters. It evaluates input by applying noise to model parameters and samples deltas based on randomness. The update method adjusts parameters using learning rate, rollouts, and standard deviation. Effective in converging to better models. |
| [hp.py](https://github.com/swarwick-dev/gymtrainer-ai/blob/main/models/hp.py) | Configuring Hyperparameters is responsible for defining and managing hyperparameter settings within the GymTrainer AI framework. Its primary function is to establish default values and allow for flexible customization through keyword arguments. The class also enables loading and saving of hyperparameter configurations, facilitating experimentation and reproducibility. |

</details>

---

##  Getting Started

###  Prerequisites

**Python**: `version 3.11.9`

###  Installation

Build the project from source:

1. Clone the gymtrainer-ai repository:
```sh
❯ git clone https://github.com/swarwick-dev/gymtrainer-ai
```

2. Navigate to the project directory:
```sh
❯ cd gymtrainer-ai
```

3. Install the required dependencies:
```sh
❯ pip install -r requirements.txt
```

###  Usage

To run the project, execute the following command:

```sh
❯ python gui/main_window.py
```

---

##  Project Roadmap

- [X] **`Task 1`**: <strike>Implement ARS modelling.</strike>
- [ ] **`Task 2`**: Improve the UI.
- [ ] **`Task 3`**: Add additional model support.

---

##  Contributing

Contributions are welcome! Here are several ways you can contribute:

- **[Report Issues](https://github.com/swarwick-dev/gymtrainer-ai/issues)**: Submit bugs found or log feature requests for the `gymtrainer-ai` project.
- **[Submit Pull Requests](https://github.com/swarwick-dev/gymtrainer-ai/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.
- **[Join the Discussions](https://github.com/swarwick-dev/gymtrainer-ai/discussions)**: Share your insights, provide feedback, or ask questions.

<details closed>
<summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your github account.
2. **Clone Locally**: Clone the forked repository to your local machine using a git client.
   ```sh
   git clone https://github.com/swarwick-dev/gymtrainer-ai
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```sh
   git checkout -b new-feature-x
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear message describing your updates.
   ```sh
   git commit -m 'Implemented new feature x.'
   ```
6. **Push to github**: Push the changes to your forked repository.
   ```sh
   git push origin new-feature-x
   ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.
8. **Review**: Once your PR is reviewed and approved, it will be merged into the main branch. Congratulations on your contribution!
</details>

<details closed>
<summary>Contributor Graph</summary>
<br>
<p align="left">
   <a href="https://github.com{/swarwick-dev/gymtrainer-ai/}graphs/contributors">
      <img src="https://contrib.rocks/image?repo=swarwick-dev/gymtrainer-ai">
   </a>
</p>
</details>

---

##  License

This project is protected under the [SELECT-A-LICENSE](https://choosealicense.com/licenses) License. For more details, refer to the [LICENSE](https://choosealicense.com/licenses/) file.

---

##  Acknowledgments

- List any resources, contributors, inspiration, etc. here.

---
