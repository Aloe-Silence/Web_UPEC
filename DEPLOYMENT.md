# Web_github Upload Package

This folder is intended for GitHub upload or cloud deployment.

## Included

- Streamlit dashboard code
- DBN service layer
- `DBN-GZ.xdsl` model
- Streamlit theme configuration
- dependency list
- example PySMILE license file

## Not Included

- No real `pysmile_license.py`
- No manuscript draft
- No cache files

## Deployment License Setup

For deployment, do not upload a completed `pysmile_license.py` file. Instead,
set one of the following environment variables in the hosting platform:

- `PYSMILE_LICENSE_B64`: recommended, base64-encoded PySMILE license bytes
- `PYSMILE_LICENSE_KEY`: comma-separated integer license key list, if required
- `PYSMILE_LICENSE`: raw PySMILE license string

The app falls back to a local `pysmile_license.py` only when neither environment
variable is available.

## Local License File Setup

If running the app on a personal computer rather than a cloud platform:

1. Copy `pysmile_license.example.py`.
2. Rename the copy to `pysmile_license.py`.
3. Open `pysmile_license.py`.
4. Find the function:

```python
def apply_pysmile_license(pysmile):
```

5. Delete the placeholder `raise RuntimeError(...)` block inside this function.
6. Paste the BayesFusion/PySMILE license call inside the function:

```python
def apply_pysmile_license(pysmile):
    pysmile.License((
        b"PASTE THE FIRST PART OF YOUR LICENSE HERE "
        b"PASTE THE NEXT PART OF YOUR LICENSE HERE "
        b"PASTE THE FINAL PART OF YOUR LICENSE HERE"
    ))
```

Do not paste the license into `pipeline_dbn_service.py`. That file already
loads the license from environment variables or from `pysmile_license.py`.

To generate the recommended Streamlit Cloud secret, run this locally after
creating `pysmile_license.py`:

```bash
python make_streamlit_secret.py
```

Then paste the printed `PYSMILE_LICENSE_B64 = "..."` line and, if shown,
`PYSMILE_LICENSE_KEY = "..."` line into the hosting platform's Secrets or
Environment Variables page.

For Streamlit Community Cloud, see `STREAMLIT_CLOUD.md`.

## Run

```bash
streamlit run pipeline_web_final.py
```
