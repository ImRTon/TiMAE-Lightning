# TiMAE & Transformer Unit Tests
This folder contains unit tests for the TiMAE and Transformer modules.  
To run the tests, execute the following commands from the root directory of the repository:  

```bash
export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)
python tests/test_transformer.py
python tests/test_ti_mae.py
```