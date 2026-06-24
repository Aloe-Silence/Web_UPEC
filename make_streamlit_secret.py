"""
Generate a Streamlit Community Cloud secret from a local PySMILE license file.

Usage:
    1. Copy pysmile_license.example.py to pysmile_license.py.
    2. Paste your real pysmile.License(...) call inside apply_pysmile_license().
    3. Run:
           python make_streamlit_secret.py
    4. Copy the printed PYSMILE_LICENSE_B64 line into Streamlit Cloud Secrets.

This script does not require importing the real PySMILE package. It captures the
license bytes from your local pysmile_license.py file.
"""

import base64


class _LicenseCapture:
    def __init__(self):
        self.license_bytes = None
        self.license_key = None

    def License(self, value, key=None):
        if isinstance(value, str):
            value = value.encode("utf-8")
        self.license_bytes = bytes(value)
        if key is not None:
            self.license_key = [int(item) for item in key]


def main():
    try:
        from pysmile_license import apply_pysmile_license
    except ImportError as exc:
        raise SystemExit(
            "Could not find pysmile_license.py. Copy pysmile_license.example.py "
            "to pysmile_license.py and add your license first."
        ) from exc

    capture = _LicenseCapture()
    apply_pysmile_license(capture)

    if not capture.license_bytes:
        raise SystemExit("No license bytes were captured from pysmile_license.py.")

    encoded = base64.b64encode(capture.license_bytes).decode("ascii")
    print('PYSMILE_LICENSE_B64 = "' + encoded + '"')
    if capture.license_key is not None:
        key_text = ",".join(str(item) for item in capture.license_key)
        print('PYSMILE_LICENSE_KEY = "' + key_text + '"')


if __name__ == "__main__":
    main()
