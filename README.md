# AlphaNet

We present **AlphaNet**, a local frame-based equivariant model designed to tackle the challenges of achieving both accurate and efficient simulations for atomistic systems.  **AlphaNet** enhances computational efficiency and accuracy by leveraging the local geometric structures of atomic environments through the construction of equivariant local frames and learnable frame transitions. And inspired by Quantum Mechanics, AlphaNet **introduces efficient multi-body message passing by using contraction of matrix product states** rather than common 2-body message passing.  Notably, AlphaNet offers one of the best trade-offs between computational efficiency and accuracy among existing models. Moreover, AlphaNet exhibits scalability across a broad spectrum of system and dataset sizes, affirming its versatility.
markdown
## Update Log (v0.1.1)

### Major Changes

1. **Jax version to speed up**
   - Provide **3x speedup** AlphaNet written in JAX, with a FLAX version and HaiKu version.
   - Users can **convert the checkpoint trained in torch to our JAX model**, we are the only one providing this
   
2. **Added tuned Ziegler–Biersack–Littmark (ZBL) potential to stablize AlphaNet**
   - We followed the ZBL theory, and fitted our own ZBL potential by calculated ~7,000 diatom systems.  
   - See [ZBL](./zbl.md) in detail

     

## Installation Guide

### Installation Steps

1. **Create a Conda Environment**

   Open your terminal or command prompt and run:

   ```bash
   conda create -n alphanet_env python=3.8 #or later version
   ```

2. **Activate the Environment**

   ```bash
   conda activate alphanet_env
   ```

3. **Install Required Packages**

   Navigate to your desired installation directory and run:

   ```bash
   pip install -r requirements.txt
   ```

4. **Clone the Repository**

   ```bash
   git clone https://github.com/zmyybc/AlphaNet.git
   ```

5. **Install AlphaNet**

   Navigate into the cloned repository and install AlphaNet in editable mode:

   ```bash
   cd AlphaNet
   pip install -e .
   ```

   This allows you to make changes to the codebase and have them reflected without reinstalling the package.

## Quick Start

### Basic Usage

The settings are put into a config file, you can see the json files provided as example, or see comments in `alphanet/config.py` for some help. 

In this version, you can set **"zbl" in the "model" field to true to enable ZBL potential**.

Our code is based on pytorch-lightning, and in this version we provide command line interaction, which makes AlphaNet easier to use. However if you are already familar with python and torch, which is not that hard, it would be great to use the model in a torch way and do further exploration. In all there are 4 commands:
1. Train a model:

```bash 
alpha-train example.json # use --help to see more functions, like multi-gpu training resuming from ckpt...
```
2. Evaluate a model and draw diagonal plot:
```bash 
alpha-eval -c example.json -m /path/to/ckpt # use --help to see more functions
```
3. Convert from lightning ckpt to state_dict ckpt:
```bash 
alpha-conv -i in.ckpt -o out.ckpt # use --help to see more functions
```
The functions above can also be used in a script way like previous version, see `old_README`.

### Make dataset
To prepare the training dataset in format of pickle, you can use:

1. from deepmd:

```bash 
python scripts/dp2pic_batch.py
```

2. from extxyz:

```bash 
python scripts/xyz2pic.py
```

So if you work in AlphaNet directory, the dataset should be organized as:
```
AlphaNet/
├── input.json
└── dataset/
    ├── my_dataset_1/ #This are your self-decided name, which should also written in your json file
    │   ├── raw/
    │   └── processed/ # would appear after you first run training, when you need to change the dataset, you should remove it
    ├── my_dataset_2/ #This are your self-decided name, which should also written in your json file
    │   ├── raw/
    │   └── processed/
    └── custom_dataset/#This are your self-decided name, which should also written in your json file
        ├── raw/
        └── processed/
```
There is also an ase calculator, you can use jax in this:

```python 
from alphanet.infer.calc import AlphaNetCalculator
from alphanet.infer.new_haiku import AlphaNetCalculator #JAX version
from alphanet.config import All_Config
from ase.build import bulk
# example usage
atoms = bulk('Cu', 'fcc', a=3.6, cubic=True)

calculator = AlphaNetCalculator(
        ckpt_path='./alex_0410.ckpt',#./pretrained/OMA/haiku/haiku_params.pkl haiku ckpt
        device = 'cuda',
        precision = '32',
        config=All_Config().from_json('./pretrained/OMA/oma.json'),
)

atoms.calc = calculator
print(atoms.get_potential_energy())

```

### Using AlphaNet in JAX
1. Installation
   ```bash
   pip install -U --pre jax jaxlib "jax-cuda12-plugin[with-cuda]" jax-cuda12-pjrt -i https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/
   ```
   This is just for reference. JAX installation may be tricky, please get more information in [JAX](https://docs.jax.dev/en/latest/installation.html) and its github issues.

   Currently I suggest **version>=0.4 <=0.4.10 or >=0.4.30 <=0.5 or >=0.6**  

   Install flax and haiku
   ```bash
   pip install flax
   pip install -U dm-haiku
   ```

2. Converted checkpoints:
   
   See pretrained/OMA directory

3. Convert a self-trained ckpt
   
   First from torch to flax:
   
   You can use scripts: scripts/torch2flax.py, what you need to modify is the config in it and the ckpt file.

   Then from flax to haiku:

   We provided a very informative script for this: scripts/flax2haiku.py, which is not a direct conversion script, but provide information about how to initialize a Flax model, get the params, save the params, load the params, convert from flax params to haiku params, initialize the haiku model, preprocess the converted haiku params, and apply it to the model. Hope this is helpful! 
     
4. Performance:
   
   The output (energy forces stress) difference from torch model would below 0.001. I ran speed tests on a 4090 GPU, system size from 4 to 300, and get a **2.5x to 3x** speed up.

   Please note jax model need to be compiled first, so the first run could take a few seconds or minutes, but would be pretty fast after that.

## Dataset Download

[The Defected Bilayer Graphene Dataset](https://zenodo.org/records/10374206)

[The Formate Decomposition on Cu Dataset](https://archive.materialscloud.org/record/2022.45)

[The Zeolite Dataset](https://doi.org/10.6084/m9.figshare.27800211)

[The OC dataset](https://opencatalystproject.org/)

[The MPtrj dataset](https://matbench-discovery.materialsproject.org/data)

## Pretrained Models

The models pretrained on **OC2M** and **MPtrj** are nearly ready for release, so you won’t have to wait much longer. Additionally, we are actively planning the release of other pretrained models in the near future.

### ​**AlphaNet-MPtrj-v1**

A new model with a small size a slight architecture change from previous one. It consists of approximately ​**4.5 million parameters**. **F1 score: 0.808**


#### ​**Access the Model**

The following resources are available in the directory:

- ​**Model Configuration**: mp.json
- ​**Model `state_dict`**: Pre-trained weights can be downloaded from [Figshare](https://ndownloader.figshare.com/files/53851133).

**Path**: `pretrained_models/MPtrj`

### ​**AlphaNet-oma-v1**

Same size with **AlphaNet-MPtrj-v1**, trained on OMAT24, and finetuned on sALEX+MPtrj. **F1 score: 0.909**


#### ​**Access the Model**

The following resources are available in the directory:

- ​**Model Configuration**: oma.json
- ​**Model `state_dict`**: Pre-trained weights can be downloaded from [Figshare](https://ndownloader.figshare.com/files/53851139).

**Path**: `pretrained_models/OMA`

## License

This project is licensed under the GNU License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

We thank all contributors and the community for their support. Please open an issue or disscusion  if there are any problems. 

## Citation
[AlphaNet: Scaling Up Local-frame-based Interatomic Potential](https://arxiv.org/abs/2501.07155)



