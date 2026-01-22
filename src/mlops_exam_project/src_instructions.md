# Pytorch instructions (Hydra)



## Running data.py


Generate processed data with download (in default directories)

```bash
uv run ./src/mlops_exam_project/data.py --download
```

Generate processed data without download

```bash
uv run ./src/mlops_exam_project/data.py --no-download
```

With custom paths

```bash
uv run ./src/mlops_exam_project/data.py --data-path ./data/raw/Wqt.csv --output-folder ./data/processed/
```


Here are examples of running the data preprocessing script with custom train-validation-split ratios:
```bash
# Example 1: Custom train/test split (70/30) with default train/val split
uv run ./src/mlops_exam_project/data.py --train-test-split-ratio 0.7

# Example 2: Custom train/val split (85/15) with default train/test split
uv run ./src/mlops_exam_project/data.py --train-val-split-ratio 0.85

# Example 3: Both custom ratios (75/25 train/test, 80/20 train/val)
uv run ./src/mlops_exam_project/data.py --train-test-split-ratio 0.75 --train-val-split-ratio 0.8

# Example 4: Custom ratios without downloading (data already exists)
uv run ./src/mlops_exam_project/data.py --no-download --train-test-split-ratio 0.9 --train-val-split-ratio 0.95

# Example 5: All custom parameters
uv run ./src/mlops_exam_project/data.py \
  --data-path ./data/raw/Wqt.csv \
  --output-folder ./data/processed/ \
  --download \
  --train-test-split-ratio 0.7 \
  --train-val-split-ratio 0.85
```
<!-- ./src/mlops_exam_project/data.py ./data/raw/Wqt.csv ./data/processed/ --download -->

## Running the training.py (with Hydra)

From the root of the project, run:

```bash
uv run src/mlops_exam_project/train.py
```

To override Hydra config values from the command line, you need to use Hydra's override syntax with ++ or standard key=value format. Here's how to do it:

```bash
uv run python src/mlops_exam_project/train.py training.epochs=2
```

Or if you want to ensure it overrides even if the key doesn't exist in the config:
```bash
uv run python src/mlops_exam_project/train.py ++training.epochs=3
```

Other examples:

```bash
# Override multiple parameters
uv run python src/mlops_exam_project/train.py training.epochs=2 training.batch_size=32 training.lr=0.0001

uv run python src/mlops_exam_project/train.py training.epochs=2 training.batch_size=32 training.lr=0.0001 "training.hidden_dims=[8,8]"

uv run python src/mlops_exam_project/train.py training.epochs=2 training.batch_size=32 training.lr=0.0001 training.hidden_dims=\[8,8\]



# Override model name
uv run python src/mlops_exam_project/train.py model_name=model_experiment1.pth

# Change config groups
uv run python src/mlops_exam_project/train.py training=default
```


## Running evaluate.py (with Hydra)


Command line examples for running the evaluation script with Hydra configuration.


### Basic usage
```bash
# Run evaluation with default configuration from config.yaml
uv run ./src/mlops_exam_project/evaluate.py

# Run evaluation from project root
uv run python src/mlops_exam_project/evaluate.py
```

### Override Configuration Parameters
```bash
# Override batch size
uv run ./src/mlops_exam_project/evaluate.py training.batch_size=64

# Override model path and name
uv run ./src/mlops_exam_project/evaluate.py model_path=models/best model_name=best_model.pth

# Override data path
uv run ./src/mlops_exam_project/evaluate.py data_path=data/processed

# Override test data filename
uv run ./src/mlops_exam_project/evaluate.py test_data_filename=custom_test_data.csv
```

### Multiple Overrides
```bash
# Override multiple parameters at once
uv run ./src/mlops_exam_project/evaluate.py \
  training.batch_size=128 \
  model_name=wine_classifier_v2.pth \
  test_data_filename=test_data.csv

# Override hidden dimensions and dropout
uv run ./src/mlops_exam_project/evaluate.py \
  training.hidden_dims=[128,64,32] \
  training.dropout_rate=0.4
```
## Running  visualize.py (with Hydra)

```bash
uv run src/mlops_exam_project/visualize.py
```
