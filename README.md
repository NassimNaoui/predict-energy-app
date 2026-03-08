# 🏨 Predicting the Energy Consumption of Seattle's Buildings

The City of Seattle aims to achieve carbon neutrality by 2050. Through this project, we propose a machine learning model capable of predicting the energy consumption of non-residential buildings to help monitor and optimize energy usage.

## 📖 Table of Contents
1. [Architecture](#architecture)
2. [Installation](#installation)
3. [Model Choice](#model-choice)
4. [BentoML Key Commands](#bentoml-key-commands)
5. [Deployment](#deployment)

---

## 🏗️ Architecture

Our MLOps pipeline and serving infrastructure are built on the following stack:

* **Data Source**: Seattle Building Energy Benchmarking Dataset
* **Processing & Serving**: Python, BentoML, and Pydantic (for strict input/output data validation)
* **Containerization**: BentoML and Docker
* **Compute & Cloud**: Google Cloud Platform (GCP) - Cloud Run

---

## 🛠️ Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management. To set up the project locally:

1. Clone the repository:
   ```bash
   git clone git@github.com:NassimNaoui/predict-energy-app.git
   cd predict-energy-app
   ````

2. Install depencies
    ```bash
   poetry install
   ````

3. Activate the virtual environment
    ```bash
   poetry shell
   ```

---

## 🌳 Model Choice

For this predictive task, we selected a **Random Forest algorithm**.
This choice is well-suited for tabular data with **potentially non-linear relationships**. It handles both numerical and categorical features effectively and provides good interpretability (feature importance) to understand **which building characteristics impact energy consumption the most**.

---

## 🍱 BentoML Key commands

Here are the essential commands to manage and test the BentoML service locally:

1. **List saved models** in your local BentoML store:
    ```bash
   bentoml models list
   ```

2. **Run the service locally** (with hot-reloading for development):
    ```bash
   bentoml serve service:svc --reload
   ```

3. **Build the Bento** (packages the model, code, and dependencies):
    ```bash
   bentoml build
   ```
4. **List built Bentos** :
    ```bash
   bentoml list
   ```

---

## 🚀 Deployment

Since development is done on an ARM architecture and GCP requires an x86_64 architecture, we use cross-compilation during the containerization step.

1. **Containerize the Bento for AMD64/x86_64**:
    ```bash
    bentoml containerize <BENTO_TAG> --platform linux/amd64
    ```

2. **Tag the Docker image for GCP Artifact Registry**:
    ```bash
    docker tag <IMAGE_ID> <REGION>-docker.pkg.dev/<PROJECT_ID>/<REPOSITORY_NAME>/<IMAGE_NAME>:<TAG>
    ```

3. **Push the image to GCP**:
    ```bash
    docker push <REGION>-docker.pkg.dev/<PROJECT_ID>/<REPOSITORY_NAME>/<IMAGE_NAME>:<TAG>
    ```

4. **Deploy to GCP Cloud Run**:
    ```bash
    gcloud run deploy <service-name> \
    --image <REGION>-docker.pkg.dev/<PROJECT_ID>/<REPOSITORY_NAME>/<IMAGE_NAME>:<TAG> \
    --platform managed \
    --region <REGION> \
    --allow-unauthenticated
    --port 3000 \
    --memory 2Gi
    ```