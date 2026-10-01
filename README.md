# Physics-guided conditional flow matching

Python code for *Physics-Guided Conditional Flow Matching for the Inversion and Uncertainty Quantification of Flows in Porous Media*.

Data and trained models are available on [Google Drive](https://drive.google.com/drive/folders/1jJCpDPurm7yHyrQLbah0TEdwTjPA53kd?usp=sharing).

The network implementations are:

- Conditional flow matching (CFM): `UNetCond3D` in [`src/module.py`](src/module.py).
- Unconditional flow matching (UFM): `SpatialCFMBackboneUFM` in [`src/ufm_module.py`](src/ufm_module.py).

The UFM network uses a 3D U-Net with time and fixed spatial encodings. It takes the permeability field and time as inputs, without conditioning on production data.

Configure file paths and sampling parameters in [`src/pipeline.py`](src/pipeline.py). The pipeline generates permeability samples, computes the corresponding production rates, and plots the results.

The default configuration selects CFM with `net_module="module"` and `net_class="UNetCond3D"`. To select UFM, use `net_module="ufm_module"` and `net_class="SpatialCFMBackboneUFM"`, and set `ckpt_path` to the matching UFM checkpoint.
