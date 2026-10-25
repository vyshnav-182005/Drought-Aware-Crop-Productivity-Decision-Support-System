"""MLflow tracking and champion registry for hurdle models."""
import os
import tempfile
from pathlib import Path

import joblib

ALIAS = "champion"


def setup():
    import mlflow
    from mlflow.tracking import MlflowClient
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        raise RuntimeError("MLFLOW_TRACKING_URI is required for MLflow operations")
    mlflow.set_tracking_uri(uri)
    return MlflowClient()


def log_and_register(district, bundle, metrics, params):
    import mlflow
    setup()
    mlflow.set_experiment("drought-weather-hurdle")
    with tempfile.TemporaryDirectory() as temporary:
        artifact = Path(temporary) / "model.joblib"
        joblib.dump(bundle, artifact)
        with mlflow.start_run(run_name=f"{district}_hurdle") as run:
            mlflow.log_params({
                "district": district,
                "horizon": params["model"]["horizon"],
                "test_start": params["split"]["test_start"],
                "probability_threshold": params["model"]["probability_threshold"],
            })
            row = metrics.iloc[0]
            mlflow.log_metrics({
                "test_r2": float(row["R2"]),
                "test_rmse": float(row["RMSE"]),
                "test_mae": float(row["MAE"]),
            })
            mlflow.log_artifact(str(artifact), artifact_path="bundle")
            run_id = run.info.run_id
    client = setup()
    model_name = f"{district}_hurdle"
    try:
        client.get_registered_model(model_name)
    except Exception:
        try:
            client.create_registered_model(
                model_name,
                description=f"Production hurdle rainfall model for {district}",
            )
        except Exception:
            # Another registration process may have created it between calls.
            client.get_registered_model(model_name)
    # MLflow 3 requires a logged-model flavor for register_model(), but this
    # service deliberately serves a portable joblib artifact.
    return client.create_model_version(
        name=model_name,
        source=f"runs:/{run_id}/bundle/model.joblib",
        run_id=run_id,
    )


def get_champion(name):
    client = setup()
    try:
        return client.get_model_version_by_alias(name, ALIAS)
    except Exception:
        versions = client.search_model_versions(f"name='{name}'")
        tagged = [version for version in versions if version.tags.get(ALIAS) == "true"]
        return max(tagged, key=lambda version: int(version.version)) if tagged else None


def set_champion(name, version):
    client = setup()
    try:
        client.set_registered_model_alias(name, ALIAS, str(version))
    except Exception as error:
        print(f"MLflow aliases unavailable; using champion tags: {error}")
    for model_version in client.search_model_versions(f"name='{name}'"):
        client.set_model_version_tag(
            name, model_version.version, ALIAS,
            "true" if str(model_version.version) == str(version) else "false",
        )


def download_champion(model_version):
    import mlflow
    setup()
    with tempfile.TemporaryDirectory() as temporary:
        path = mlflow.artifacts.download_artifacts(
            artifact_uri=f"runs:/{model_version.run_id}/bundle", dst_path=temporary
        )
        return joblib.load(Path(path) / "model.joblib")


def champion_metric(model_version, key):
    client = setup()
    return client.get_run(model_version.run_id).data.metrics.get(key)
