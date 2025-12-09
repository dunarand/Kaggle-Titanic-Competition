# Kaggle Titanic Competition

This repository was created for showcasing my work on the [Kaggle Competition: Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic).

## Installation

You can clone the repository to explore the notebooks on your own.

```bash
git clone https://github.com/dunarand/Kaggle-Titanic-Competition
cd Kaggle-Titanic-Competition
```

Then, create a Python virtual environment.

```bash
python3 -m venv ./.venv
```

Activate the Python environment. On Windows, run the following command in PowerShell

```PowerShell
.\.venv\Scripts\Activate.ps1
```
*(If you encounter execution errors, try running `Set-ExecutionPolicy RemoteSigned -Scope Process` first.)*

On Linux/macOS, run

```bash
source .venv/bin/activate
```

Then, install the required packages:

```bash
pip3 install -r requirements.txt
```

Next up, obtain the datasets `train.csv` and `test.csv` through the [official competition webpage](https://www.kaggle.com/competitions/titanic/data) or by using the Kaggle API:

```bash
kaggle competitions download -c titanic
```

**IMPORTANT:** Ensure you have your `kaggle.json` API token placed in the correct directory (`~/.kaggle/` on Linux/macOS or `C:\Users\<user>\.kaggle/` on Windows). You can generate this file from your [Kaggle Account Settings](https://www.kaggle.com/settings) under the API section.

Unzip the contents into the `./data/` directory.

```bash
mkdir -p data/
unzip titanic.zip -d data/
rm titanic.zip
```

Next, launch a local Jupyter server:

```bash
jupyter notebook
```

You can now work with the notebooks on your local machine.

## Project Structure

Notebooks are located in the `notebooks\` subdirectory.

- [0 - Titanic Machine Learning from Disaster](./notebooks/0%20-%20Titanic%20Machine%20Learning%20from%20Disaster.ipynb)
- [1 - Understanding & Planning](./notebooks/1%20-%20Understanding%20&%20Planning.ipynb)
- [2 - Data Wrangling](./notebooks/2%20-%20Data%20Wrangling.ipynb)
- [3 - Exploratory Data Analysis](./notebooks/3%20-%20Exploratory%20Data%20Analysis.ipynb)
- [4 - Building a Model](./notebooks/4%20-%20Building%20a%20Model.ipynb)
- [5 - Results](./notebooks/5%20-%20Results.ipynb)
