# AlphaNet-MPtrj-v1

A model trained on the MpTrj dataset.

## Model Details

* **Parameters:** Approximately 4.5 million

## Access the Model

The following resources are available in the `pretrained_models/MPtrj` path:

* **Model Configuration:** `mp.json`
* **Model state\_dict:** Pre-trained weights can be downloaded from [Figshare](https://ndownloader.figshare.com/files/53851133).

## Performance on WBM Test Set

The detailed evaluation metrics for the model on the `full_test_set` are as follows:

| Metric | Value | Unit/Description |
| :--- | :--- | :--- |
| F1 | 0.789 | fraction |
| DAF | 4.312 | dimensionless |
| Precision | 0.74 | fraction |
| Recall | 0.846 | fraction |
| Accuracy | 0.923 | fraction |
| TPR | 0.846 | fraction |
| FPR | 0.062 | fraction |
| TNR | 0.938 | fraction |
| FNR | 0.154 | fraction |
| TP | 37311.0 | count |
| FP | 13119.0 | count |
| TN | 199752.0 | count |
| FN | 6781.0 | count |
| MAE | 0.04 | eV/atom |
| RMSE | 0.091 | eV/atom |
| R2 | 0.747 | dimensionless |