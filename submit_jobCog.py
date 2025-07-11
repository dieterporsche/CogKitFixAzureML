# submit_job.py
from pathlib import Path

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, Input, Output, command
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.ai.ml.entities import Environment

# ------------------------------------------------------------------
# 1) Azure-ML-Verbindung
# ------------------------------------------------------------------
SUB_ID = "898ebc3c-57d6-4f37-b363-18025f16ef18"
RG_NAME = "CloudCityManagedServices"
WS_NAME = "playground"

ml_client = MLClient(DefaultAzureCredential(), SUB_ID, RG_NAME, WS_NAME)

# ------------------------------------------------------------------
# 2) Umgebung (falls nötig hochzählen)
# ------------------------------------------------------------------
env_name, env_version = "CogVideo", "22"
env = Environment(
    name=env_name,
    version=env_version,
    description="CogVideo Environment",
    image="mcr.microsoft.com/azureml/curated/acpt-pytorch-2.2-cuda12.1:latest",
    conda_file=Path(__file__).with_name("CogVideo.yml"),
)
# ml_client.environments.create_or_update(env)  # nur bei neuer Version

# ------------------------------------------------------------------
# 3) Assets: Daten- und Modell-Inputs, Output-Binding
# ------------------------------------------------------------------
train_data = Input(
    path="azureml:CogVideoTest:1",
    type=AssetTypes.URI_FOLDER,
    mode=InputOutputModes.RO_MOUNT,
)

# Lokaler Modell-Ordner wird als Input hochgeladen und gemountet
# local_model = Input(
#     path="/mnt/models/CogVideoX1.5-5B-I2V",
#     type=AssetTypes.URI_FOLDER,
#     mode=InputOutputModes.RO_MOUNT,
# )

run_output = Output(type=AssetTypes.URI_FOLDER, mode="upload")

# ------------------------------------------------------------------
# 4) Job-Definition
# ------------------------------------------------------------------
job = command(
    code="/home/azureuser/cloudfiles/code/Users/dieter.holstein/runs/HuggingFace/CogKitFix",
    command="""
        echo OUTPUT1
        pwd
        ls
        sudo mkdir -p CogVideoX1.5-5B-I2V
        echo OUTPUT2
        pwd
        ls
        sudo chown -R $USER:$USER CogVideoX1.5-5B-I2V
        echo OUTPUT3
        pwd
        ls
        
        huggingface-cli download THUDM/CogVideoX1.5-5B-I2V --local-dir CogVideoX1.5-5B-I2V
        echo OUTPUT4
        pwd
        ls
        export CODE_DIR=$(pwd)
        export MODEL_PATH="CogVideoX1.5-5B-I2V"   
        export DATA_ROOT="${{inputs.train_data}}"
        export OUTPUT_DIR=$CODE_DIR/output
        mkdir -p "$OUTPUT_DIR"
        echo OUTPUT5
        pwd
        ls
        
        python quickstart/scripts/i2v/iterative_training.py \
          --learning-rate 5.0e-6 \
          --batch-size 4 \
          --epochs 5

        cp -R "$OUTPUT_DIR"/. "${{outputs.run_output}}"/
    """,
    inputs={
        "train_data": train_data,
        #"local_model": local_model,
    },
    outputs={
        "run_output": run_output,
    },
    environment=f"{env_name}:{env_version}",
    compute="Cluster-NC24ads-1",
    experiment_name="dieter_CogVideo_Training",
    display_name="train_CogVideo_lora_job",
    environment_variables={
        "BNB_CUDA_VERSION": "124",
    },
)

# ------------------------------------------------------------------
# 5) Job abschicken
# ------------------------------------------------------------------
print("Submitting job …")
returned_job = ml_client.create_or_update(job)
print("Job submitted:", returned_job.name)
