# Working with Luigi Workflows

This project shows how Luigi workflows are used as pipelines for data science. A well-formed dataset is used to ensure the data science itself doesn't interfere with how Luigi works.

## How it Works

There are two Python files contained within the `workflows` folder that contain all the code. The first workflow (`workflow_one`) just shows how Luigi works with a very simple example. The real workflow is contained within `workflow_two` and is the genesis of this project.

The basic notion, from a data science perspective, is to take a corpus, vectorize it, split it into test and train sets, pickle it (for later use), then use logistic regression to build a predictive model.

The dataset used is from Reddit.

## Running It
To see the entire pipeline run with PDF outputs from Luigi, run `workflwo_two.py`.



