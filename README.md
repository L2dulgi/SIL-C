# SIL-C (Final Implementation: LazySI)

This repository contains the codebase for the project referred to as **SIL-C** in the associated paper. During its development, the project was known under various names including **Apposi**, **Assil**, and **LazySI**. While these earlier names represent different stages of the project, the **final implementation is named LazySI**, and the term **SIL-C** is used for publication and documentation purposes.

---

## Installation

1. **Set up the Python environment** (using Conda recommended):

   ```bash
   conda create -n silc python==3.10.16
   conda activate silc
   pip install -r requirements.txt
   pip install -e .
   ```

2. **Install MuJoCo** and set required environment variables before running experiments:

   Ensure that **mujoco210** is installed in your home directory, then add these lines to your shell configuration (`.bashrc`, `.zshrc`, etc.) or export them manually:

   ```bash
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/yourworkingdirectory/.mujoco/mujoco210/bin
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
   export MUJOCO_GL=egl
   export XLA_PYTHON_CLIENT_PREALLOCATE=false
   ```

---

## Datasets

1. **Download Datasets and Unzip**

Follow the steps below to download the dataset and extract it into the `data/` directory:

```bash
mkdir data
cd data
gdown 'https://drive.google.com/uc?id=1DbSFIUgt_Ys0l4988VXshE50z7IWL_Kq'
unzip evolving_datasets.zip
```

> This will download and unzip `evolving_datasets.zip`, which contains the necessary dataset files for training and evaluation.

2. **Verify Extracted Content**

After extraction, ensure that the following directories or files are present (example):

```bash
ls
evolving_world/
evolving_kitchen/
```

---


## Remote Environment Setup

### Kitchen (D4RL)

Set up the evaluation environment for the Kitchen domain:

```bash
conda create -n kitchen_eval python==3.8.18
conda activate kitchen_eval
pip install -r ./remoteEnv/kitchen/requirements.txt
```

> This creates a dedicated Conda environment named `kitchen_eval` with Python 3.8.18 and installs all necessary dependencies for the Kitchen (D4RL) environment.

---

### Metaworld

Set up the environment for the multi-stage Metaworld tasks:

```bash
pip install gdown
gdown 'https://drive.google.com/uc?id=1BzWk9vbJIaEkklfeA0F2C8ncRJCoPHcz'
unzip mmworld.zip -d ./remoteEnv/multiStageMetaworld/
```

> This creates a new environment `mmworld_eval`, downloads the `mmworld.zip` package containing the environment code, and extracts it to the appropriate directory.

Then, set up the environment:

```bash
conda create -n mmworld_eval python==3.10.16
conda activate mmworld_eval
cd ./remoteEnv/multiStageMetaworld/mmworld
bash env.sh
pip install -e .
```

> This runs the setup script `env.sh` to install any additional dependencies and installs the environment in editable mode so changes to the codebase take effect immediately.

## Running Experiments

1. **Start the environment server**:

   ```bash
   conda activate {env_name}_eval
   python remoteEnv/{env_name}/{env_name}_server.py
   ```

2. **Launch training(in a different shell)**:

   ```bash
   bash exp/scripts/example.sh
   ```

   Additional command examples can be found in `exp/scripts/example.sh`.


---

## Logs and Saved Artifacts

Experiment results are stored in the following directory structure:

```
logs/{env_name}/{scenario_name}/sync/{baselines}/.../{date}seed0{id}/
├── policy/
│   └── policy_{version}/pre_{skill_decoder_and_interface_phase}.pkl
└── skills/
    ├── decoder_pre_{sil_phase}.pkl
    └── interface_pre_{sil_phase}.pkl
```

---

## Evaluating Results

To evaluate experiment outputs, use:

```bash
python src/AppOSI/utills/llmetrics.py -e {environment} -g {keyword1} {keyword2}
```

---

## Acknowledgements

This project includes components adapted from the [Metaworld] repository, which is distributed under the MIT License.

