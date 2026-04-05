# AlphaNet Documentation

## User Documentation

- [README.md](../README.md) - Project overview and quick start
- [mliap_lammps.md](mliap_lammps.md) - LAMMPS integration guide
- [zbl.md](zbl.md) - ZBL potential documentation

## Example Configurations

- [config_basic.json](examples/config_basic.json) - Basic training config
- [config_advanced.json](examples/config_advanced.json) - Advanced config (with ZBL)
- [config_finetune.json](examples/config_finetune.json) - Finetuning config
- [quickstart.py](examples/quickstart.py) - Quick start script

## Development Docs

- [CONTRIBUTING.md](../CONTRIBUTING.md) - Contribution guide
- [alphanet/config.py](../alphanet/config.py) - Configuration parameters

## Code Structure

```
alphanet/
├── __init__.py
├── cli.py              # CLI interface
├── config.py           # Configuration management
├── train.py            # Training logic
├── models/             # Model definitions
│   ├── alphanet.py     # Core network architecture
│   ├── model.py        # Model wrapper
│   └── ...
├── data/               # Data loading
├── infer/              # Inference tools
│   ├── calc.py         # ASE calculator
│   └── lammps_mliap_alphanet.py  # LAMMPS interface
└── ...
```

## FAQ

See [README.md#FAQ](../README.md#FAQ)
