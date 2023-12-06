# Deep-Reinforcement-Learning-Chess-AI

pip install torch-related lib using following cmd:

`pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio===0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`

After installing the torch-related lib, use following cmd to install other remaining libraries:

`pip install -r requirements.txt`

To train single agent, use the following cmd:

`python train.py --agent single`

To train double agents, use the following cmd:

`python train.py --agent double`

### Refrences
- https://doi.org/10.5281/zenodo.7789509

### Python Environment 
- Python 3.9
