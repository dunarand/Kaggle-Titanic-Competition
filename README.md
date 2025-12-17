# Kaggle Titanic Competition

## Introduction

This repository was created for showcasing my work on the
[Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic).

## Project Scope

This project aims to develop a machine learning model for the Kaggle Titanic
competition while sticking to data science principles. I not only aimed for
a model built by my best effort, I also worked on creating a repository that
would be up to par with industry standards in terms of data science workflow.

My primary goal was to showcase my work principle rather than building the best
model possible. My logic was that it is possible that the model I've built will
be obsolete due to newer and better ways to achieve the same goal of predicting
whether a passenger survived or not. However, by sticking to the modern data
science principles, following a logical flow, and exercising ethical, grounded,
and maintainable practices, the repository will remain relevant for years to
come.

Hence, this project contains 5 main notebooks, each for the respective data
science project stage. We start by understanding the problem and conducting an
initial plan. We then perform data wrangling, EDA, building and evaluating a
model, and we share the results. You can read more about the scope and the
goals we've set for this project on the
[main entry notebook](./notebooks/0%20-%20Titanic%20Machine%20Learning%20from%20Disaster.ipynb)

## Results

TODO

# Installation

You can clone the repository and work on your local machine.

## Requirements

- Python version >= 3.11.9 (older versions not tested)
- Python packages in `requirements.txt` (older versions not tested)

## Installation Steps

You can clone the repository to explore the notebooks on your own.

```bash
git clone https://github.com/dunarand/Kaggle-Titanic-Competition
cd Kaggle-Titanic-Competition
```

Then, create a Python virtual environment.

```bash
python3 -m venv ./.venv
```

Activate the Python environment. On Windows, run the following command in
PowerShell

```PowerShell
.\.venv\Scripts\Activate.ps1
```

*(If you encounter execution errors, try running
`Set-ExecutionPolicy RemoteSigned -Scope Process` first.)*

On Linux/macOS, run

```bash
source .venv/bin/activate
```

Then, install the required packages:

```bash
pip3 install -r requirements.txt
```

**Note:** This repository was created using
[neovim](https://github.com/neovim/neovim) for a better editing experience, and
[jupytext](https://github.com/mwouts/jupytext) for a better version control
system. If you plan to work with the same tools, I highly suggest the
[jupytext.nvim plugin](https://github.com/GCBallesteros/jupytext.nvim).
If you want to work on the `ipynb` Jupyter Notebooks with your choice of IDE or
text editor, you will not need the `jupytext` installation in
`requirements.txt`, so you can remove the respective line.

Next up, obtain the datasets `train.csv` and `test.csv` through the
[official webpage](https://www.kaggle.com/competitions/titanic/data) or by
using the Kaggle API:

```bash
kaggle competitions download -c titanic
```

**IMPORTANT:** Ensure you have your `kaggle.json` API token placed in the
correct directory (`~/.kaggle/` on Linux/macOS or `C:\Users\<user>\.kaggle\`
on Windows). You can generate this file from your
[Kaggle Account Settings](https://www.kaggle.com/settings) under the API
section.

Unzip the contents into the `./data/raw` directory.

```bash
mkdir -p data/
unzip titanic.zip -d data/raw
rm titanic.zip
```

Next, launch a local Jupyter server:

```bash
jupyter notebook
```

You can now work with the notebooks on your local machine.

## Project Structure

Notebooks are located in the `notebooks/` subdirectory.

- [0 - Titanic Machine Learning from Disaster](./notebooks/0%20-%20Titanic%20Machine%20Learning%20from%20Disaster.ipynb)
- [1 - Understanding & Planning](./notebooks/1%20-%20Understanding%20&%20Planning.ipynb)
- [2 - Data Wrangling](./notebooks/2%20-%20Data%20Wrangling.ipynb)
- [3 - Exploratory Data Analysis](./notebooks/3%20-%20Exploratory%20Data%20Analysis.ipynb)
- [4 - Building a Model](./notebooks/4%20-%20Building%20a%20Model.ipynb)
- [5 - Results](./notebooks/5%20-%20Results.ipynb)

You can find the pickled models in the `models/` subdirectory.
