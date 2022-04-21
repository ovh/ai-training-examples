---
title: AI Notebooks - Tutorial - Create your first AI notebook using Miniconda image
slug: notebooks/tuto-first-ai-notebook
excerpt: How to build your first Machine Learning model?
section: AI Notebooks tutorials
order: 7
---

**Last updated 21th April, 2022.**

## Objective

This tutorial will allow you to create your first **OVHcloud AI notebook** based on a very simple Machine Learning model: the **simple linear regression**.

At the end of this tutorial, you will have learned to master OVHcloud AI Notebooks and be able to predict the scores obtained by students as a function of the number of hours worked.

![image](images/linear-regression-student.png){.thumbnail}

## Requirements

- Access to the [OVHcloud Control Panel](https://www.ovh.com/auth/?action=gotomanager&from=https://www.ovh.co.uk/&ovhSubsidiary=GB)
- An AI Notebooks project created inside a [Public Cloud project](https://www.ovhcloud.com/en-gb/public-cloud/) in your OVHcloud account
- A user for AI Notebooks
- The downloaded `student_scores.csv` file: [Students Score Dataset](https://www.kaggle.com/datasets/shubham47/students-score-dataset-linear-regression)

> [!primary]
>
> In this tutorial, you will be able to predict a student's exam score based on the amount of time he has studied using a dataset available on [Kaggle](https://www.kaggle.com/): [Students Score Dataset](https://www.kaggle.com/datasets/shubham47/students-score-dataset-linear-regression). Download this `student_scores.csv` file before following this tutorial.
>

## Instructions

### Uploading your dataset on Public Cloud Storage

If you want to upload `student_scores.csv` file from the [OVHcloud Control Panel](https://www.ovh.com/auth/?action=gotomanager&from=https://www.ovh.co.uk/&ovhSubsidiary=GB), go to the Object Storage section and [create a new object container](https://docs.ovh.com/gb/en/storage/pcs/create-container/) by clicking `Object Storage`{.action} > `Create an object container`{.action}.

![image](images/new-object-container.png){.thumbnail}

If you want to run it with the CLI, just follow this [this guide](https://docs.ovh.com/gb/en/publiccloud/ai/cli/access-object-storage-data/). You have to choose the region, the name of your container and the path where your data is located and use the following command:

``` {.console}
ovhai data upload <region> <container> student_scores.csv
```

### Launching and accessing Jupyter notebook with "Miniconda" framework

To launch your notebook from the [OVHcloud Control Panel](https://www.ovh.com/auth/?action=gotomanager&from=https://www.ovh.co.uk/&ovhSubsidiary=GB), refer to the following steps.

Otherwise, go to the **Command lines in the ovhai CLI** part.

#### Code editor

Choose the `Jupyterlab` code editor.

#### Framework

In this tutorial, the `Miniconda` framework is used.

> [!warning]
>
> With **Miniconda**, you will be able to set up your environment by installing the Python libraries you need.
>

You can choose the version of `conda` you want.

> [!primary]
>
> The default version of `conda` is functional for this tutorial.
>

#### Attach your data

- Select `Attach an Object storage Container`.

You need to attach a volume with the `student_scores.csv` file from the OVHcloud Object Storage. For more information on data, volumes and permissions, see [our guide on data](https://docs.ovh.com/gb/en/publiccloud/ai/cli/access-object-storage-data/).

To be able to use the source code below in this article you have to create an Object Storage containers mounted as follows:

- storage container: container name which contains the `student_scores.csv` file
- mount point name: `/workspace/data`
- permission: `read only`

#### Attach the GitHub repository

- Select `Attach a public Git repository`

To access and test the notebook, clone the GitHub `ovh/ai-training-examples` repository.

- git repository URL: https://github.com/ovh/ai-training-examples.git
- mount point name: `/workspace/ai-training-examples`
- permission: `read write`

#### Command lines in the ovhai CLI

If you want to launch it with the CLI, choose the `jupyterlab` editor and the `conda` framework.

To access the different versions of `conda` available, run the following command.

``` {.console}
ovhai capabilities framework list -o yaml
```

> [!primary]
>
> If you do not specify a version, your notebook starts with the default version of `conda`.
>

You can attach your data from the Object Storage and the GitHub repository by following this [documentation](https://docs.ovh.com/gb/en/publiccloud/ai/cli/access-object-storage-data/).

Choose the number of GPUs (`<nb-gpus>`) to use in your notebook and use the following command.

``` {.console}
ovhai notebook run conda jupyterlab \
	--name <notebook-name> \
	--framework-version <conda-version> \
  --volume <container>@<region>/:/workspace/data:RO \
  --volume https://github.com/ovh/ai-training-examples.git:/workspace/ai-training-examples:RW \
  --gpu <nb-gpus>
```

You can then reach your notebook’s URL once the notebook is running.

### Access to the notebook

Once the repository has been cloned, find your notebook by following this path: `ai-training-examples` > `notebooks` > `miniconda` > `tuto` > `notebook-introduction-linear-regression.ipynb`.

A preview of this notebook can be found on GitHub [here](https://github.com/ovh/ai-training-examples/blob/first-notebook-miniconda/notebooks/miniconda/tuto/ai-notebooks-introduction/notebook-introduction-linear-regression.ipynb).

## Feedback

Please send us your questions, feedback and suggestions to improve the service:

- On the OVHcloud [Discord server](https://discord.com/invite/vXVurFfwe9)
