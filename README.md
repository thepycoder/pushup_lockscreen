# Pushup Lockscreen

## First setup

`pip install clearml`
and
`clearml-init`

## Getting the data

### ClearML data setup

First create a dataset, we'll periodically update it when we label new data.
To create our dataset we just have to run

```
clearml-data create --project <PROJECT_NAME> --name <DATASET_NAME>
```

Add the PROJECT_NAME and DATASET_NAME variables you chose to the config file located at <INPUT REQUIRED>
as well as the dataset ID