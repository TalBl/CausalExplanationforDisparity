# CausalExplanationforDisparity

we introduce DisEx, a framework designed to generate causal explanations for disparities between two groups of interest.

## Table of Contents
- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)

---

## Getting Started

This repository contains code for running **Causal Explanations for Disparity**, which is located in the `algorithms/final_algorithm/full.py` script. The following instructions will guide you on how to set up and execute the project locally.

---

## Prerequisites

Ensure that you have the following installed on your system:
- Python 3.7 or later
- A package manager, such as `pip`

---

## Installation

1. Clone the repository:

2. Install dependencies:
bash
```pip install -r requirements.txt```

## Usage

Navigate to the algorithms/final_algorithm/ directory:

bash
``` cd algorithms/final_algorithm ```
Run the main script:
bash
```python full.py```

### Using Your Own Dataset
If you want to test the algorithm with your own dataset:

Build a suitable Dataset object. You can find examples of how to structure the dataset in the full.py file.

To define the two groups for comparison:

Add two columns to your dataset named group1 and group2.
Each column should contain values of 0 or 1:
* 1 indicates that a row belongs to the respective group.
* 0 indicates that it does not. <br>
<br>Ensure that your dataset is properly formatted and aligns with these requirements before running the script.
