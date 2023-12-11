# Deep-Reinforcement-Learning-Chess-AI

![chessPic](https://github.com/MiuMiuMiue/Deep-Reinforcement-Learning-Chess-AI/assets/81593292/8e1670f1-4ce2-4a79-ba79-905404d6d3a7)

# Jupyter Notebook
There is a notebook `betaChess.ipynb` provided in the github. We have tested the notebook on Colab, it can install the specific Python version and all the libraries needed for the project. Therefore, it is better to run the notebook on Colab as it will prepare the environment.

# Environment Set up
Since we are using specific Torch libraries, python 3.9 is required to install the torch packages. 

pip install torch-related lib using following cmd:

`pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio===0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`

After installing the torch-related lib, use following cmd to install other remaining libraries:

`pip install -r requirements.txt`

# Training the Models

To train single agent, use the following cmd:

`python train.py --agent single`

To train double agents, use the following cmd:

`python train.py --agent double`

Other parameters, such as resume training from checkpoints, are also provided. Details are in the train.py.

Directory `model_ckpts` and `renders` need to be created manually in order to save the checkpoints and demo videos in both directories `results/DoubleAgents` and `results/SingleAgent`.

# Evaluating the Models
`python playGames.py white-ckpt () black-ckpt ()`

white-ckpt and black-ckpt are needed to indicate which models to use for each side in the games.

# Generating Plots
`python plots.py`

This will generate Learning Curves for both Single Agent Training and Double Agent Training according to existing data in the .npy files.

# Checkpoints
The checkpoints for the models that mentioned in the report can be downloaded here: https://drive.google.com/drive/folders/11IsrpM9x2eYo3CxUvz2K9xNAys7-OPrm?usp=sharing

# Refrences
- https://doi.org/10.5281/zenodo.7789509

