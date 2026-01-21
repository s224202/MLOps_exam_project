


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



