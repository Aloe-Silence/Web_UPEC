# Web-Enabled DBN Platform for Underground Pipeline External Corrosion

This repository contains a Streamlit implementation of a web-enabled dynamic
Bayesian network (DBN) platform for probabilistic assessment of underground
pipeline external corrosion.

The platform provides two interactive workspaces:

- **Baseline Evolution**: configure X1-X20 evidence and predict Low,
  Moderate, and High corrosion-depth probabilities across selected time slices.
- **Scenario Explorer**: hold the baseline evidence fixed, vary one selected
  factor, compare all time-slice probability distributions, and summarize TSW
  changes relative to the baseline.

## Repository Contents

- `pipeline_web_final.py`: Streamlit dashboard.
- `pipeline_dbn_service.py`: PySMILE service layer for DBN inference.
- `DBN-GZ.xdsl`: GeNIe/SMILE DBN model used by the web application.
- `pysmile_linux/pysmile.so`: BayesFusion PySMILE Linux binary for
  Streamlit Cloud deployment with Python 3.12.
- `.streamlit/config.toml`: Streamlit theme configuration.
- `pysmile_license.example.py`: template for local PySMILE license setup.
- `STREAMLIT_CLOUD.md`: step-by-step Streamlit Community Cloud deployment guide.
- `make_streamlit_secret.py`: helper for generating Streamlit Cloud secrets.

## License Requirement

This project depends on BayesFusion PySMILE for inference. The PySMILE license
is not redistributed in this repository. Obtain your own valid PySMILE license
from BayesFusion before running or deploying the app.

For local execution, copy the example file:

```bash
copy pysmile_license.example.py pysmile_license.py
```

Then edit `pysmile_license.py`. The license code should be placed inside the
`apply_pysmile_license(pysmile)` function in that file. Specifically, delete the
placeholder `raise RuntimeError(...)` block and paste your own
`pysmile.License(...)` call at the same indentation level.

Example structure:

```python
def apply_pysmile_license(pysmile):
    pysmile.License((
        b"PASTE THE FIRST PART OF YOUR LICENSE HERE "
        b"PASTE THE NEXT PART OF YOUR LICENSE HERE "
        b"PASTE THE FINAL PART OF YOUR LICENSE HERE"
    ))
```

Do not paste the real license into `pipeline_dbn_service.py`. The service layer
will automatically call `apply_pysmile_license(pysmile)` when the model is
loaded. The real `pysmile_license.py` file is excluded by `.gitignore`.

For deployment, configure the license as an environment variable instead of
committing a license file. The app checks these variables first:

- `PYSMILE_LICENSE_B64`: base64-encoded PySMILE license bytes. This is the
  recommended deployment option because it is robust to spaces and line breaks.
- `PYSMILE_LICENSE_KEY`: comma-separated integer license key list, if required
  by your BayesFusion license.
- `PYSMILE_LICENSE`: raw PySMILE license string.

If neither environment variable is set, the app falls back to the local
`pysmile_license.py` file.

For public cloud deployment, do not upload `pysmile_license.py`. Use the hosting
platform's **Secrets** or **Environment Variables** page and set
`PYSMILE_LICENSE_B64` and `PYSMILE_LICENSE_KEY` there.

To generate the recommended secret values from a local `pysmile_license.py`
file, run:

```bash
python make_streamlit_secret.py
```

The script prints the `PYSMILE_LICENSE_B64 = "..."` line and, when required by
the license, the `PYSMILE_LICENSE_KEY = "..."` line. Paste those lines into
Streamlit Cloud Secrets.

For Streamlit Community Cloud deployment, see `STREAMLIT_CLOUD.md`.

## Installation

Create and activate a Python environment, then install the open-source
dependencies:

```bash
pip install -r requirements.txt
```

For Streamlit Cloud deployment, the repository includes the BayesFusion PySMILE
Linux binary for Python 3.12 under `pysmile_linux/pysmile.so`. The service layer
automatically loads this file if `import pysmile` is not available. Do not
install the unrelated PyPI package named `pysmile`; it is not the Bayesian
network inference library used by this app.

## Run The App

```bash
streamlit run pipeline_web_final.py
```

The app uses `DBN-GZ.xdsl` by default. Each time slice represents two years in
the current implementation.

## Notes For Reproducibility

- The web application performs DBN inference locally through PySMILE.
- The Transitional Susceptibility Window (TSW) is identified as the interval in
  which the Moderate corrosion-depth probability is greater than or equal to
  both Low and High probabilities.
- Scenario analysis follows a one-factor-at-a-time design: one selected factor
  is varied while the remaining input evidence is held fixed.
