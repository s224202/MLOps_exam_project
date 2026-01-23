


# Docker instructions (with Hydra)


Note that since our .src files utilizy, we must  *train_hydra.dockerfile* and *evaluate_hydra.dockerfile*.


First  we explain how to use docker image built using
```bash
train_hydra.dockerfile
```

## Build the training image

From the root of the project, run:

```bash
docker build -f dockerfiles/train_hydra.dockerfile . -t {image_name}:{image_tag}
```

Example:

```bash
docker build -f dockerfiles/train_hydra.dockerfile . -t train_hydra:latest
```


Without caching:
```bash
docker build  --no-cache -f dockerfiles/train_hydra.dockerfile . -t train_hydra:latest
```


## Run the training container


### with default arguments
```bash
docker run --name {container_name} name:tag
```
Example:
```bash
docker run --name experiment_hydra1 train_hydra:latest
```




## Extract outputs from the container
Extract the model:
```bash
docker run --rm \
  --name experiment_hydra1 \
  -v $(pwd)/models:/models \
  train_hydra:latest \
  training.epochs=2
```

<!-- docker run --rm  --name experiment_hydra1 train_hydra:latest \
-v $(pwd)/models:/models/ train:latest -->

<!-- ```bash
docker run  -v $(pwd)/models:/models/  --name experiment_hydra1 train_hydra:latest
``` -->

Extract figures:
```bash
docker run --rm \
  --name experiment_hydra1 \
  -v $(pwd)/reports/figures:/reports/figures  \
  train_hydra:latest \
  training.epochs=2
```
<!-- docker run --name experiment1 -v $(pwd)/reports/figures:/reports/figures train:latest --epochs 1 -->



Extract models and figures:


```bash
docker run --rm \
  --name experiment_hydra1 \
  -v $(pwd)/models:/models \
  -v $(pwd)/reports/figures:/reports/figures  \
  train_hydra:latest \
  training.epochs=2

```

<!-- docker run  -v $(pwd)/models:/models/ \
  -v $(pwd)/models:/models \
            -v $(pwd)/reports/figures:/reports/figures \
            --name experiment3 train:latest --epochs 1 -->


## Passing Hydra overrides via docker run

Hydra overrides work exactly like CLI flags.

```bash
docker run --rm --name experiment_hydra1 \
  train_hydra:latest \
  training.epochs=5
 ```

##  Mounting configs from the host (recommended)


Here we explain how to use docker image built using
```bash
train_hydra_config_mounting.dockerfile
```
Example
```bash
docker build -f dockerfiles/train_hydra_config_mounting.dockerfile . -t train_hydra:latest
```



This lets you edit configs without rebuilding the image.

```bash
docker run --rm \
   -v $(pwd)/configs:/configs \
  -v $(pwd)/models:/models \
  -v $(pwd)/reports/figures:/reports/figures  \
    train_hydra:latest \
  training.epochs=5
  ```

## Build the evaluation image

From the root of the project, run:

```bash
docker build -f dockerfiles/evaluate_hydra.dockerfile . -t {image_name}:{image_tag}
```

Example:

```bash
docker build -f dockerfiles/evaluate_hydra.dockerfile . -t evaluate:latest
```




### Run evaluation (mounting model + data)

Example:   Mount the file to a file (most common)
```bash
docker run --rm --name evaluate \
  -v $(pwd)/configs:/configs \
  -v $(pwd)/models/model.pth:/models/model.pth \
  -v $(pwd)/data:/data \
  evaluate:latest \
  model_path=/models \
  model_name=model.pth
```

Example: Mount the whole models directory (Useful if you have multiple checkpoints)
```bash
docker run --rm --name evaluate \
  -v $(pwd)/configs:/configs \
  -v $(pwd)/models:/models \
  -v $(pwd)/data:/data \
  evaluate:latest \
  model_path=/models \
  model_name=model.pth
```






# Docker instructions (without Hydra)

## Build the training image

From the root of the project, run:

```bash
docker build -f dockerfiles/train.dockerfile . -t {image_name}:{image_tag}
```

Example:

```bash
docker build -f dockerfiles/train.dockerfile . -t train:latest
```



## Run the training container

### with specified arguments
```bash
docker run --name {container_name} train:latest --lr 1e-3 --batch-size 16 --epochs 1
```
Example:
```bash
docker run --name experiment1 train:latest --lr 1e-3 --batch-size 16 --epochs 1
```

### with default arguments
```bash
docker run --name {container_name} train:latest
```
Example:
```bash
docker run --name experiment1 train:latest
```



## Extract outputs from the container
Extract the model:
```bash
docker run --name experiment1 -v $(pwd)/models:/models/ train:latest
```

```bash
docker run  -v $(pwd)/models:/models/  --name experiment3 train:latest --epochs 1
```

Extract figures:
```bash

docker run --name experiment1 -v $(pwd)/reports/figures:/reports/figures train:latest --epochs 1
```


Extract models and figures:


```bash
docker run  -v $(pwd)/models:/models/ \
            -v $(pwd)/reports/figures:/reports/figures \
            --name experiment3 train:latest --epochs 1
```


Clean up:
```bash
docker rm experiment3
```
or using the ID:
```bash
docker --rm <ID>
```

or automatically delete the container after it exits by including `--rm`:
```bash
docker run --rm --name experiment1 train:latest
```
or
```bash
docker run  --rm -v $(pwd)/models:/models/ \
            -v $(pwd)/reports/figures:/reports/figures \
            --name experiment3 train:latest --epochs 1
```

If it might be running, force-remove:
```bash
docker rm -f experiment3
```





## Build the evaluation image

From the root of the project, run:

```bash
docker build -f dockerfiles/evaluate.dockerfile . -t {image_name}:{image_tag}
```

Example:

```bash
docker build -f dockerfiles/evaluate.dockerfile . -t evaluate:latest
```

### Run evaluation (mounting model + data)

Example:   Mount the file to a file (most common)
```bash
docker run --rm --name evaluate \
    -v $(pwd)/models/model.pth:/models/model.pth \
    -v $(pwd)/data:/data \
    evaluate:latest \
    /models/model.pth
```

Example: Mount the whole models directory (Useful if you have multiple checkpoints)
```bash
docker run --rm --name evaluate \
    -v $(pwd)/models:/models \
    -v $(pwd)/data:/data \
    evaluate:latest \
    /models/model.pth
```
```
