# Deploy on Streamlit Community Cloud

This guide is for deploying the interactive Streamlit app and obtaining a
public `streamlit.app` URL.

## Important PySMILE Note

This project uses **BayesFusion PySMILE**. Do not install the unrelated PyPI
package named `pysmile`; that package is for the SMILE data format and is not
the Bayesian network inference library used here.

Before deployment, confirm that BayesFusion PySMILE can be installed in the
Streamlit Community Cloud Linux environment. If BayesFusion provides a Linux
wheel or installation instruction for your Python version, add that installation
method to `requirements.txt` according to BayesFusion's guidance.

## Files To Upload

Upload the contents of this `Web_github` folder to a GitHub repository. The
repository root should contain:

- `pipeline_web_final.py`
- `pipeline_dbn_service.py`
- `DBN-GZ.xdsl`
- `requirements.txt`
- `.streamlit/config.toml`
- `.gitignore`
- documentation files

Do not upload a completed `pysmile_license.py` file.

## Generate The Secret

On your local computer, after creating a working `pysmile_license.py`, run:

```bash
python make_streamlit_secret.py
```

The script prints one or two lines like:

```toml
PYSMILE_LICENSE_B64 = "PASTE_GENERATED_VALUE_HERE"
PYSMILE_LICENSE_KEY = "PASTE_GENERATED_VALUE_HERE"
```

Copy these lines. Do not paste them into GitHub.

## Streamlit Cloud Steps

1. Go to `https://share.streamlit.io`.
2. Click **Create app**.
3. Select your GitHub repository.
4. Select the branch, usually `main`.
5. Set the main file path to:

```text
pipeline_web_final.py
```

6. Open **Advanced settings**.
7. Select a Python version compatible with your BayesFusion PySMILE wheel.
8. In **Secrets**, paste the generated lines:

```toml
PYSMILE_LICENSE_B64 = "PASTE_GENERATED_VALUE_HERE"
PYSMILE_LICENSE_KEY = "PASTE_GENERATED_VALUE_HERE"
```

9. Click **Deploy**.

After deployment, Streamlit provides a public URL similar to:

```text
https://your-project.streamlit.app
```

## If Deployment Fails

The most likely cause is PySMILE installation, not the Streamlit code. Check the
build log for errors such as:

- `ModuleNotFoundError: No module named 'pysmile'`
- incompatible Python version
- unsupported Linux wheel

If Streamlit Community Cloud cannot install BayesFusion PySMILE, use a server or
a Docker-capable platform such as Render, Railway, or a university server.
