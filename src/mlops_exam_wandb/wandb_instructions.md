# Weights  \& Biases  instructions (with Hydra)


##  Set-up

Add  relevant W&B information  the .env file  (*WANDB_API_KEY*, *WANDB_PROJECT* and *WANDB_ENTITY* are mandatory), such as 

```bash
WANDB_API_KEY=<WANDB_API_KEY>
WANDB_PROJECT=<WANDB_PROJECT>
WANDB_ENTITY=<WANDB_ENTITY>
WANDB_JOB_TYPE=<WANDB_JOB_TYPE>
```

*Example:*
```bash
WANDB_API_KEY=<WANDB_API_KEY>
WANDB_PROJECT=mlops_exam_project
WANDB_ENTITY=mr-mikael-sorensen 
WANDB_JOB_TYPE=training
```


##  connecting to wandb account

```bash
uv run wandb login
```

## Running train_wandb.py

Run the training script with default configuration:
 ```bash
 uv run python src/mlops_exam_wandb/train_wandb.py  
```


Run the training script while overriding parameters.

 *Example*:  Here the number of training epochs  is set to 2:
```bash
 uv run python src/mlops_exam_wandb/train_wandb.py training.epochs=2
```
 *Example*:   Here the number of training epochs  is set to 2 and a  custom hidden layer dimensions (8, 8) using quoted list syntax

```bash
uv run python src/mlops_exam_wandb/train_wandb.py training.epochs=2 training.hidden_dims='[8,8]'
```

 *Example*:  Here the number of training epochs  is set to 2 and a  custom hidden layer dimensions (8, 8) using escaped brackets
```bash
 uv run python src/mlops_exam_wandb/train_wandb.py training.epochs=2 training.hidden_dims=\[8,8\]
```


## Weights & Biases agent sweep

Here is a three-step procedure on how to use the Weights & Biases agent sweep.


1. Update exisiting *sweep.yaml* file, if necessary. Alternatively create a new  *sweep.yaml* file. As an example
```bash
program: src/mlops_exam_wandb/train_wandb.py
name: wine_quality_sweep
project: mlops_exam_project  
entity: mr-mikael-sorensen  
method:  random #bayes
metric:
    goal: minimize
    name: val_loss
parameters:
    #lr:
    training.lr:
        min: 0.0001
        max: 0.1
        distribution: log_uniform_values # does not work bayes
    #hidden_dims:
    training.hidden_dims:
        values: [[8, 8], [16, 8], [16, 16], [32, 16]]  
    #dropout_rate:
    training.dropout_rate:
        min: 0.0
        max: 0.15
        distribution: uniform
    # batch_size:
    #     values: [16, 32, 64]
    # epochs:
    #     values: [5, 10, 15]
run_cap: 10
command:
  - ${env}
  - python
  - ${program}
  - ${args_no_hyphens}

```

2. Initialize and run the sweep

```bash
uv run wandb sweep configs/sweep.yaml
```
This will output a sweep ID, denoted <SWEEP_ID>. 

3. start an agent to run the sweep:

```bash
uv run wandb agent WANDB_ENTITY/WANDB_PROJECT/<SWEEP_ID>
```
<!-- WANDB_PROJECT=mlops_exam_project
WANDB_ENTITY=mr-mikael-sorensen #my_entity

```bash
uv run wandb agent mr-mikael-sorensen/mlops_exam_project/<SWEEP_ID>
``` -->

*Example*:
```bash
uv run wandb agent mr-mikael-sorensen/mlops_exam_project/64kudfpr
```
Alternatively, we can run the sweep using the command:

```bash
uv run wandb agent <SWEEP_ID>
```
*Example*:
```bash
uv run wandb agent 64kudfpr
```


##  Load a model from the W&B artifact registry

Model from the W&B artifact registry can be loaded as follows
```bash
uv run python src/mlops_exam_wandb/load_model_from_artifact.py
```
Examples with specified artifact, project, entity:
```bash
uv run src/mlops_exam_wandb/load_model_from_artifact.py --artifact-name "red_wine_quality_model:v17"
```
or  
```bash
 uv run src/mlops_exam_wandb/load_model_from_artifact.py --artifact-name "red_wine_quality_model:v17" --project "mlops_exam_project" --entity "mr-mikael-sorensen"
```

Or in the  code:
```bash
from load_model_from_artifact import load_model_from_artifact

model = load_model_from_artifact("red_wine_quality_model:v2")
```


 




##  Load a model from the W&B model registry

Model from the W&B model registr  registry can for instance  be loaded as follows (here the default settinge will be used):
```bash
uv run python src/mlops_exam_wandb/load_model_from_registry.py
```

 
## Evaluate a  model

A model  can be evaluated  use it in three ways.

1.  Local checkpoint (default):

Example:
```bash
uv run python src/mlops_exam_wandb/evaluate_wandb.py
```
2. W&B Artifact:

Example:

```bash
uv run python src/mlops_exam_wandb/evaluate_wandb.py use_wandb_artifact=true wandb_artifact_name=red_wine_quality_model:latest  
+wandb_artifact_name=red_wine_quality_model:latest
```
or 

```bash
uv run python src/mlops_exam_wandb/evaluate_wandb.py ++use_wandb_artifact=true ++wandb_artifact_name=red_wine_quality_model:latest
```

<!-- ```bash
uv run python src/mlops_exam_wandb/evaluate_wandb.py use_wandb_artifact=true wandb_artifact_name=red_wine_quality_model:v5
``` -->

3. W&B Registry:

Example:
```bash
uv run python src/mlops_exam_wandb/evaluate_wandb.py use_wandb_registry=true wandb_registry_model_name=red_wine_quality_model wandb_registry_version=latest
```

or with a specific version:

```bash
uv run python src/mlops_exam_wandb/evaluate_wandb.py use_wandb_registry=true wandb_registry_model_name=red_wine_quality_model wandb_registry_version=v0
```