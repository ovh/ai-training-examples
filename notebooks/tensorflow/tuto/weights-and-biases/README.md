*Last updated 9th of September, 2021.*


## Objective

The purpose of this tutorial is to show how it is possible to use **Weights & Biases**, one of the most famous Developer tool for machine learning, with OVHcloud AI Notebooks.

Weight and Biases allow you to track your machine learning experiments, version your datasets and manage your models easily, like shown below :

![image](images/overview_wandb.png)

This tutorial presents two examples of using Weights & Biases. The first notebook will use the TensorFlow image.

## Requirements

- access to the [OVHcloud Control Panel](https://www.ovh.com/auth/?action=gotomanager);
- a Public Cloud project created;
- a Public Cloud user with the ability to start AI Notebooks;
- a Weights & Biases account,  you can create it on their [website](https://wandb.ai/site). It's Free for individuals.

## Instructions

### Launch and access a Jupyter notebook

The first step will consist in creating a Jupyter Notebook with OVHcloud AI Notebooks.

First, you have to install the OVHAI CLI then just choose the name of the notebook (`<notebook-name>`) and the number of GPUs (`<nb-gpus>`) to use on your job and use the following command:

- TensorFlow image docker:

```bash
ovhai notebook run tensorflow jupyterlab \
    --name <notebook-name> \
    --gpu <nb-gpus>
```

Whatever the selected method, you should now be able to reach your notebook’s URL.

### Clone the GitHub examples repository

The GitHub repository containing all examples for OVHcloud AI NOTEBOOKS is available [here](https://github.com/ovh/ai-training-examples).

Inside your notebook, open a new Terminal tab by clicking `File` > `New` > `Terminal`.

![image](images/new-terminal.png)

Run the following command in the notebook’s terminal to clone the repository:

```bash
git clone https://github.com/ovh/ai-training-examples.git
```

### Experiment with OVHcloud examples notebooks

Once the repository has been cloned, find the notebook of your choice.

- The notebook using TensorFlow and Weights & Biases is based on the MNIST dataset. To access it, follow this path:

`ai-training-examples` > `notebooks` > `tensorflow` > `tuto` > `notebook_Weights_and_Biases_MNIST.ipynb`

Instructions are directly shown inside the notebooks. You can run them with the standard "Play" button inside the notebook interface.

#### Notebook using TensorFlow and Weights & Biases is based on the MNIST dataset

The aim of this tutorial is to show how it is possible, thanks to Weights & Biases, to compare the results of trainings according to the chosen hyperparameters.

For example, you can display the accuracy and loss curves for your valid and train data. These metrics will be displayed for each epoch of each training.

![image](images/valid_train_metrics_mnist_wandb.png)

You can then compare your trainings using the **Parallel coordinates** graph type:

![image](images/parallel_coordinates_mnist_wandb.png)

You can also compare the **Test error rates**:

![image](images/test_error_rate_mnist_wandb.png)

## Conclusion

To sum up, **Weights & Biases** allows you to quickly track your experiments, version and iterate data sets, evaluate model performance, reproduce models, visualise results and spot regressions, and share results with your colleagues.

You can use it directly on OVHcloud AI Notebooks in few minutes.
