# file name does not start with test_, so pytest will not run it automatically, this is intentional

import random

from locust import HttpUser, task, between


class WineQualityUser(HttpUser):
    wait_time = between(1, 3)  # Wait between 1 and 3 seconds between tasks

    @task
    def classify_wine(self):
        # Generate random wine features
        features = {
            "fixed_acidity": random.uniform(4.0, 15.0),
            "volatile_acidity": random.uniform(0.1, 1.5),
            "citric_acid": random.uniform(0.0, 1.0),
            "residual_sugar": random.uniform(1.0, 15.0),
            "chlorides": random.uniform(0.01, 0.2),
            "free_sulfur_dioxide": random.uniform(5, 50),
            "total_sulfur_dioxide": random.uniform(15, 200),
            "density": random.uniform(0.9900, 1.0050),
            "pH": random.uniform(2.5, 4.0),
            "sulphates": random.uniform(0.3, 2.0),
            "alcohol": random.uniform(8.0, 15.0),
        }
        # Send POST request to the API
        self.client.post("/predict", json=features)

    @task
    def health_check(self):
        self.client.get("/")
