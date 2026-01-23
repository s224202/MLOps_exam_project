from fastapi import FastAPI
from http import HTTPStatus
from typing import Any
from loguru import logger

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
                input_dim=11,
                hidden_dims=[64, 32],
                output_dim=10,
                dropout_rate=0.2,
            )
            try:
                # convert features to tensor
                import torch
                import numpy as np

                feature_array = np.array(
                    [features[key] for key in sorted(features.keys())],
                    dtype=np.float32,
                ).reshape(1, -1)
                feature_tensor = torch.from_numpy(feature_array)
                raw = model(feature_tensor)
            except Exception as e:
                # model likely expects tensors; fall back to dummy
                logger.error(f"Error processing features: {e}")
                raw = None

            # make result JSON serializable
            if raw is None:
                prediction = {"quality": 5}
                logger.info("Using dummy prediction.")
            else:
                try:
                    # torch tensors -> list
                    import torch

                    if isinstance(raw, torch.Tensor):
                        prediction = raw.detach().cpu().tolist()
                    else:
                        prediction = raw
                    logger.info(f"Prediction: {prediction}")
                except Exception:
                    prediction = raw
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            prediction = {"quality": 5}

    response = {
        "message": HTTPStatus.OK.phrase,
        "status_code": HTTPStatus.OK.value,
        "prediction": prediction,
    }
    return response


@app.post("/predict_onnx")
async def predict_onnx(features: dict):
    """Return a model prediction for the provided features using ONNX runtime.

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
            import onnxruntime as ort
            import numpy as np

            # Load the ONNX model
            ort_session = ort.InferenceSession("models/onnx_wine_model.onnx")

            # Prepare input data
            feature_array = np.array(
                [features[key] for key in sorted(features.keys())],
                dtype=np.float32,
            ).reshape(1, -1)

            # Run inference
            ort_inputs = {ort_session.get_inputs()[0].name: feature_array}
            ort_outs = ort_session.run(None, ort_inputs)

            # Make result JSON serializable
            prediction = ort_outs[0].tolist()
            logger.info(f"ONNX Prediction: {prediction}")
        except Exception as e:
            logger.error(f"Error during ONNX prediction: {e}")
            prediction = {"quality": 5}

    response = {
        "message": HTTPStatus.OK.phrase,
        "status_code": HTTPStatus.OK.value,
        "prediction": prediction,
    }
    return response
