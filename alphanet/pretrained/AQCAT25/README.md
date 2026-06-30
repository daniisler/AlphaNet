# AlphaNet-AQCAT25

A model trained on the [AQCAT25](https://www.sandboxaq.com/aqcat25), mainly used for surfaces adsorbtion and reactions. The model is trained on the total energies and forces of trainiing set and slabs data.

## Model Details

* **Parameters:** Approximately 6.9M

## Access the Model

The following resources are available in the `pretrained_models/AQCAT25` path:
* **Model Configuration:** `aqcat.json`
* **Model state\_dict:** Pre-trained weights `aqcat_1021.ckpt`


## Performance 
| Mae | Value | Unit/Description |
| :--- | :--- | :--- |
| test_id | 0.010,0.088 | eV/atom , eV/$ \AA $|
| test_ood_ads | 0.010,0.082 | eV/atom , eV/$ \AA $ |
| test_ood_both | 0.024, 0.097 | eV/atom , eV/$ \AA $ |
| test_ood_mat | 0.0186, 0.101 | eV/atom , eV/$ \AA $ |
| test_ood_slabs | 0.025, 0.091 | eV/atom , eV/$ \AA $ |
