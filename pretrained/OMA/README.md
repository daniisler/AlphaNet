# AlphaNet-oma-v1

Same size with **AlphaNet-MPtrj-v1**, trained on OMAT24, and finetuned on sALEX+MPtrj. **F1 score: 0.909**

## Performance on WBM full\_test\_set

| Metric | Value | Unit/Description |
| :--- | :--- | :--- |
| F1 | 0.892 | fraction |
| DAF | 5.022 | dimensionless |
| Precision | 0.862 | fraction |
| Recall | 0.924 | fraction |
| Accuracy | 0.961 | fraction |
| TPR | 0.924 | fraction |
| FPR | 0.031 | fraction |
| TNR | 0.969 | fraction |
| FNR | 0.076 | fraction |
| TP | 40735.0 | count |
| FP | 6539.0 | count |
| TN | 206332.0 | count |
| FN | 3357.0 | count |
| MAE | 0.019 | eV/atom |
| RMSE | 0.066 | eV/atom |
| R2 | 0.865 | dimensionless |

## Access the Model

The following resources are available in the directory:

* **Model Configuration**: `oma.json`
* **Model `state_dict`**: [alex_1212.ckpt](./alex_1212.ckpt).

**Path**: `pretrained_models/OMA`
