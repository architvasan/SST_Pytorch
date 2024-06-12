# SST_Pytorch
## Set up conda environment:
```
conda env create -f my_conda.yaml
```

## Run inference:

```python
python sst_reg_infer_mpi.py -m MODELWEIGHTS -L DATALOCATION -D DATASET -s SMILESCOLUMN -c MODELCONFIG -b BATCH
```
Where:
* `MODELWEIGHTS`: saved weights for the model
* `DATALOCATION`: path where all compound datasets are stored
* `DATASET`: which dataset you want to target (small test: BDB)
* `SMILESCOLUMN`: column for SMILES data (default: SMILE)
* `MODELCONFIG`: json file for model config (default: config_mod.json)
* `BATCH`: batch size for inference (default: 128)
