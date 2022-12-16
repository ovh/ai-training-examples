# Examples of use of OVHcloud AI Solutions

![githubbanner](https://user-images.githubusercontent.com/3379410/27423240-3f944bc4-5731-11e7-87bb-3ff603aff8a7.png)

[![Maintenance](https://img.shields.io/maintenance/yes/2022.svg)]() [![Chat on gitter](https://img.shields.io/gitter/room/ovh/ux.svg)](https://gitter.im/ovh/ai)

This repository centralize all resources and examples (such as notebooks) that could be use with the [OVHcloud AI Training product](https://www.ovhcloud.com/en-gb/public-cloud/ai-training/)

# Installation

## Get the sources

```bash
git clone https://github.com/ovh/ai-training-examples.git

cd ai-training-examples
```

## Read and experiment the tutorials

The tutorials are categorised by **product** and by **task**.

We offer examples on how to use **AI Notebooks**, **AI Training** and **AI Apps**. There are many forms: `python` files, `ipython` notebooks, `Dockerfile`, ...

The tutorials structure is as follows:

```bash
.
├── apps
│   ├── fastapi
│   │   └── spam-classifier-api
│   ├── flask
│   │   ├── hello-world
│   │   ├── object-detection-yolov5-app
│   │   │   ├── static
│   │   │   └── templates
│   │   └── sentiment-analysis-hugging-face-app
│   │       ├── static
│   │       └── templates
│   ├── getting-started
│   │   └── flask
│   │       └── hello-world-api
│   ├── gradio
│   │   └── sketch-recognition
│   └── streamlit
│       ├── audio-classification-app
│       ├── eda-classification-iris
│       ├── simple-app
│       └── speech-to-text
├── jobs
│   ├── jupyterlab
│   │   └── tensorflow
│   └── weights-and-biases
│       └── audio-classification-models-comparaison
│           ├── data-processing
│           └── models-training
└── notebooks
    ├── audio
    │   └── audio-classification
    ├── computer-vision
    │   ├── image-classification
    │   │   └── tensorflow
    │   │       ├── resnet50
    │   │       ├── tensorboard
    │   │       └── weights-and-biases
    │   └── object-detection
    │       └── miniconda
    │           ├── yolov5
    │           │   └── weights-and-biases
    │           └── yolov7
    │               └── images
    ├── getting-started
    │   ├── miniconda
    │   │   └── ai-notebooks-introduction
    │   ├── pytorch
    │   └── tensorflow
    └── natural-language-processing
        ├── speech-to-text
        │   └── miniconda
        │       ├── advanced
        │       │   └── sounds
        │       ├── basics
        │       │   └── sounds
        │       └── compare-models
        │           └── sounds
        └── text-classification
            ├── hugging-face
            │   └── sentiment-analysis-twitter
            │       ├── BARThez
            │       ├── BERT
            │       └── CamemBERT
            └── miniconda
                └── spam-classifier
```

# Related links

 * Documentation: https://docs.ovh.com/gb/en/publiccloud/ai/
 * Contribute: https://github.com/ovh/ai-training-examples/blob/master/CONTRIBUTING.md
 * Report bugs: https://github.com/ovh/ai-training-examples/issues

# License

Copyright 2021 OVH SAS

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
