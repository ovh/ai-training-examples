# Compare 2 methods by running 2 jobs in parallel in order to classify sounds. See which one performs best on your data!

## Requirements 

- a DockerHub account 
- a Weights and Biases account

> :warning: **WANDB API KEY**: After cloning the repo, please, make sure to replace `MY_WANDB_API_KEY` by yours in the two Python files for training.

- `train-image-classification-audio-files-csv.py`
- `train-image-classification-audio-files-spectrograms.py`

## Download your data

First, download the data on [Kaggle](https://www.kaggle.com/datasets/subhajournal/free-spoken-digit-database). 

It's a zip file (`audio_files.zip`)! Push it into an object container named `spoken-digit`.

You can unzip it.

You should have:

```console
├── spoken-digit
    └── audio_files.zip
    └── audio_files
        └── zero
        └── one
        └── two
        └── three
        └── four
        └── five
        └── six
        └── seven
        └── eight
        └── nine
```

> :warning: Make sure to go to the **right directory** before building your Docker image!
> 

```console
cd ai-training-examples/jobs/weights-and-biases/audio-classification-models-comparaison
```

## Build and push the Docker image

Please, replace `your_docker_id` by yours!

```console
docker build . -t your_docker_id/two-models:latest
```

```console
docker push your_docker_id/two-models:latest
```

## Data processing

Currently, you can use the following methods.

### Audio to csv with features extraction

```console
ovhai job run --cpu 12 --volume spoken-digit@GRA/:/workspace/data:RW:cache your_docker_id/two-models:latest -- bash -c 'python data-processing/data-processing-audio-files-csv.py'
```

### Audio to spectrogram with image generation

```console
ovhai job run --cpu 12 --volume spoken-digit@GRA/:/workspace/data:RW:cache your_docker_id/two-models:latest -- bash -c 'python data-processing/data-processing-audio-files-spectrograms.py'
```

You should have your object storage:

```console
├── spoken-digit
    └── audio_files.zip
    └── audio_files
        └── zero
        └── one
        └── ...
        └── nine
    └── csv_files
        └── data_3_sec.csv
    └── spectrograms
        └── zero
        └── one
        └── ...
        └── nine
    └── spectrograms_split
        └── train
            └── zero
            └── one
            └── ...
            └── nine
        └── oval
            └── zero
            └── one
            └── ...
            └── nine
```

## Training

Launch 2 jobs with AI Training:

### ANN for audio classification based on audios feature

```console
ovhai job run --gpu 1 --volume spoken-digit@GRA/:/workspace/data:RO:cache your_docker_id/two-models:latest -- bash -c 'python models-training/train-classification-audio_files_csv.py'
```

### CNN for image classification based on spectrograms

```console
ovhai job run --gpu 1 --volume spoken-digit@GRA/:/workspace/data:RO:cache your_docker_id/two-models:latest -- bash -c 'python models-training/train-image-classification-audio-files-spectrograms.py'
```

### Weights & Biases

To check your models training, please run:

```console
ovhai job logs <job_id>
```

Before the training starts, an URL will appear to access to `WANDB`. Copy/past it on you web browser.

> :warning: Your jobs have to be in `RUNNING` status to check the logs!

To get the job status:

```console
ovhai job get <job_id>
```
