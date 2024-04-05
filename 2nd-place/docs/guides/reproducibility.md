This guide contains steps necessary to reproduce the competition results.

Before running any of the commands in this section, please make sure you have configured your local development
environment by following [this](setup-dev-env.md) guide.

You have two options for reproducing the results:

1. Running from scratch
2. Using model checkpoints and generated dataset files

## I want to train from scratch

If you want to fully reproduce the solution results starting from raw training data. Please follow this steps.

### Preparing the data

To prepare the data follow steps outlined in this section.

#### Download the data

Download and extract the data to following directories in the root of the repo:

```
kelp-wanted-competition/
└── data
   └── raw
      ├── train
      │   ├── images             <= place training images here
      │   └── masks              <= place training masks here
      ├── test
      │   └── images             <= place test images here
      └── metadata_fTq0l2T.csv   <= place the metadata file directly in the `raw` dir
```

Run in order:

* Plot samples - for better understanding of the data and quick visual inspection

    ```shell
    make sample-plotting
    ```

* AOI Grouping - will group the similar images into AOIs and use those groups to generate CV-folds.

    ```shell
    make aoi-grouping
    ```

* EDA - run Exploratory Data Analysis to visualize statistical distributions of different image features

    ```shell
    make eda
    ```

* Calculate band statistics - will calculate per-band min, max, mean, std etc. statistics (including spectral indices)

    ```shell
    make calculate-band-stats
    ```

* Train-Val-Test split with Stratified K-Fold Cross Validation

    ```shell
    make train-val-test-split-cv
    ```

The generated `train_val_test_dataset_strategy=cross_val.parquet` metadata lookup file
and `YYYY-MM-DD-Thh:mm:ss-stats-fill_value=nan-mask_using_qa=True-mask_using_water_mask=True.json`
files will have to be used as inputs for the training scripts, both locally and for Azure ML Pipelines.

### Training the models

For model training you have two options. Train them on Azure ML or train them locally.

#### Via Azure ML (AML)

> Note: You'll need Azure Subscription and Azure DevOps Organization for that one.
> You'll also need a basic knowledge of Azure services such as Entra ID, Blob Storage and Azure ML.

1. Create Azure ML Workspace
2. Setup Service Principal with access to the Azure ML Workspace
3. Setup Azure DevOps variable group
(see [.env-sample](https://github.com/xultaeculcis/kelp-wanted-competition/blob/main/.env-sample) for what variables are needed)
4. Setup Service Connections for GitHub, AML Workspace and ARM Resource Group (use SP created earlier for it)
5. Setup Azure DevOps Pipelines
6. In Azure ML set up the following:
   1. [Datasets datastore](https://github.com/xultaeculcis/kelp-wanted-competition/blob/main/aml/resources/datastores/datasets-datastore.yaml)
   2. Training Dataset Data Asset - please upload training data to Blob Storage and register it as Folder Asset
   3. Dataset Stats Data Asset - please upload the stats file to Blob Storage and register it as File Asset
   4. Dataset Metadata Data Asset - please upload the metadata parquet file (generated by `train-val-test-split-cv` Makefile command)
   to Blob Storage and register it as File Asset
   5. [Compute Clusters with spot instances](https://github.com/xultaeculcis/kelp-wanted-competition/tree/main/aml/resources/compute)
   6. [Training Environment](https://github.com/xultaeculcis/kelp-wanted-competition/tree/main/aml/environments/acpt_train_env)
7. Once done you'll need to modify the versions and names in the
[AML components](https://github.com/xultaeculcis/kelp-wanted-competition/tree/main/aml/components)
and [AML pipelines](https://github.com/xultaeculcis/kelp-wanted-competition/tree/main/aml/pipelines) to match the resource
names you have just created. I recommend you use
[Azure ML CLI](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli?view=azureml-api-2&tabs=public)
to set them up from the terminal.
See `yaml` files in the [aml](https://github.com/xultaeculcis/kelp-wanted-competition/tree/main/aml) folder for details.
8. You can now trigger the Azure ML Hyperparameter Search or Model Training Pipelines via Azure DevOps Pipelines

#### Locally

* Run all folds training:

    ```shell
    make train-all-folds
    ```

* Run single fold training:

    ```shell
    make train FOLD_NUMBER=<fold-number>
    ```

See the Makefile definition to see all available options.

> Note: Both Azure ML Pipelines and Makefile commands have been adjusted to use hyperparameters used in the best model submission.

### Making predictions

Once the models have been trained you can generate submission file with them by adjusting the run_dir paths in the
Makefile and running following commands:

#### Single model

Run:

```shell
make predict-and-submit
```

The submission directory will be created under `data/submissions/single-model`.

#### Ensemble

Running ensemble prediction is just as easy with following command:

```shell
make cv-predict
```

The submission directories will be created under `data/submissions/avg`.

This will run prediction with each fold individually and then average the predictions using weights specified in the
Makefile. The weights and decision thresholds in the Makefile commands are already set up to be in line with the winning
ones. You just need to adjust the fold dir paths.

> Note: Please note that all folds were used for the best submissions. You'll need to train all folds!

## I want to use model checkpoints

If you want to just reproduce the final submissions without running everything from scratch follow steps in this section.

### Download models and dataset

Download following files.

#### Download checkpoints
The full model training run directories are hosted here:

* Best single model (private LB 0.7264): [best-single-model.zip](https://1drv.ms/u/s!Agzv09pDi6JJhjoJqtKQh0w0Yp-w?e=TOhaDg)
* Best submission #1 (private LB 0.7318): [top-submission-1.zip](https://1drv.ms/u/s!Agzv09pDi6JJhj0M-Mq0wKpGNpOl?e=gTbCRQ)
* Best submission #2 (private LB 0.7318): [top-submission-2.zip](https://1drv.ms/u/s!Agzv09pDi6JJhj69ORSRdT9gicqP?e=FDxKSZ)

Please download them and extract to `models` directory. The final directory structure should look like this:

```
models/
├── best-single-model
│   └── Job_sincere_tangelo_dm0xsbhc_OutputsAndLogs
├── top-submission-1
│   ├── Job_elated_atemoya_31s98pwg_OutputsAndLogs
│   ├── Job_hungry_loquat_qkrw2n2p_OutputsAndLogs
│   ├── Job_icy_market_4l11bvw2_OutputsAndLogs
│   ├── Job_keen_evening_3xnlbrsr_OutputsAndLogs
│   ├── Job_model_training_exp_65_OutputsAndLogs
│   ├── Job_model_training_exp_67_OutputsAndLogs
│   ├── Job_nice_cheetah_grnc5x72_OutputsAndLogs
│   ├── Job_strong_door_yrq9zpmd_OutputsAndLogs
│   ├── Job_willing_pin_72ss6cnc_OutputsAndLogs
│   └── Job_yellow_evening_cmy9cnv7_OutputsAndLogs
└── top-submission-2
    ├── Job_boring_foot_hb224t08_OutputsAndLogs
    ├── Job_coral_lion_x39ft9cb_OutputsAndLogs
    ├── Job_dreamy_nut_fkwzmgxh_OutputsAndLogs
    ├── Job_icy_airport_7r8h9q3c_OutputsAndLogs
    ├── Job_lemon_drop_cxncbygc_OutputsAndLogs
    ├── Job_loving_insect_hvd7v5p9_OutputsAndLogs
    ├── Job_plum_angle_0f163gk5_OutputsAndLogs
    ├── Job_plum_kettle_36dw15zk_OutputsAndLogs
    ├── Job_tender_foot_07bt1687_OutputsAndLogs
    └── Job_wheat_tongue_mjzjpvjw_OutputsAndLogs
```

#### Download competition data

Download the competition data as described in [Download the Data](#download-the-data) section.

#### Download prepped files

If you don't want to run all data preparation steps you can just download the metadata and dataset stats files from here:

* Dataset metadata - train, val and test 10-fold CV split:
[train_val_test_dataset.parquet](https://1drv.ms/u/s!Agzv09pDi6JJhjvhAJsn5QHuYUMo?e=V1J7Ll) - place it in
`data/processed` directory
* Dataset per-band statistics: [ds-stats.zip](https://1drv.ms/u/s!Agzv09pDi6JJhjwozX54HRhgmYzw?e=BSC51e) - extract it
and place the JSON file in `data/processed` directory

The final `data` directory should have the following structure:

```
data/
├── auxiliary
├── interim
├── predictions                <= single model predictions from ensembles will be saved here
├── processed
│   ├── 2023-12-31T20:30:39-stats-fill_value=nan-mask_using_qa=True-mask_using_water_mask=True.json
│   └── train_val_test_dataset.parquet
├── raw
│   ├── train
│   │   ├── images             <= place training images here
│   │   └── masks              <= place training masks here
│   ├── test
│   │   └── images             <= place test images here
│   └── metadata_fTq0l2T.csv   <= place the metadata file directly in the `raw` dir
└── submissions
    ├── avg                    <= ensemble submissions will be saved here
    └── single-model           <= single model submissions will be saved here
```

### Making predictions

To reproduce the best submissions follow this steps.

> NOTE: The Makefile commands expect the model and dataset files to be in correct directories!

#### Single model

Run:

```shell
make repro-best-single-model-submission
```

The submission directory will be created under `data/submissions/single-model`.

#### Ensemble

To reproduce top #1 submission with Priv LB score of 0.7318 run:

```shell
make repro-top-1-submission
```

To reproduce top #2 submission with Priv LB score of 0.7318 run:

```shell
make repro-top-2-submission
```

The submission directories will be created under `data/submissions/avg`. Individual model predictions will be saved
under `data/predictions/top-1-submission` and `data/predictions/top-2-submission`.