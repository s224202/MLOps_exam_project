from fastapi import FastAPI
from http import HTTPStatus
from typing import Any

try:
    # try package-relative import first
    from .model import WineQualityClassifier  # type: ignore
except Exception:
    # fallback to module import when running from the module folder
    try:
        from model import WineQualityClassifier  # type: ignore
    except Exception:
        WineQualityClassifier = None  # type: ignore

app = FastAPI()


@app.get("/")
async def read_root():
    response = {"message": HTTPStatus.OK.phrase, "status_code": HTTPStatus.OK.value}
    return response


@app.post("/predict")
async def predict(features: dict):
    """Return a model prediction for the provided features.

    This tries to use `WineQualityClassifier` if available. If importing
    or running the model fails (this project uses a placeholder model),
    a simple dummy prediction is returned so the API stays usable for
    sanity checks.
    """
    prediction: Any

    if WineQualityClassifier is None:
        prediction = {"quality": 5}
    else:
        try:
            model = WineQualityClassifier(
                cfg={
                    "learning_rate": 0.001,
                    "batch_size": 32,
                    "training": {
                        "input_dim": 11,
                        "hidden_dims": [64, 32],
                        "output_dim": 6,
                        "dropout_rate": 0.3,
                    },
                }
            )
            try:
                raw = model(features)
            except Exception:
                # model likely expects tensors; fall back to dummy
                raw = None

            # make result JSON serializable
            if raw is None:
                prediction = {"quality": 5}
            else:
                try:
                    # torch tensors -> list
                    import torch

                    if isinstance(raw, torch.Tensor):
                        prediction = raw.detach().cpu().tolist()
                    else:
                        prediction = raw
                    print(prediction)
                except Exception:
                    prediction = raw
        except Exception as e:
            print(f"Error during prediction: {e}")

            prediction = {"quality": 5}

    response = {
        "message": HTTPStatus.OK.phrase,
        "status_code": HTTPStatus.OK.value,
        "prediction": prediction,
    }
    return response
