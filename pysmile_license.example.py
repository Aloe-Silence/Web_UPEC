"""
Example local PySMILE license configuration.

Copy this file to ``pysmile_license.py`` and replace the placeholder with your
own BayesFusion/PySMILE license initialization. Do not commit the real
``pysmile_license.py`` file to a public repository.
"""


def apply_pysmile_license(pysmile):
    # Put your BayesFusion/PySMILE license inside this function.
    #
    # Step 1: Copy this file and rename it:
    #         pysmile_license.example.py -> pysmile_license.py
    #
    # Step 2: Delete the RuntimeError block below.
    #
    # Step 3: Paste your own pysmile.License(...) call here.
    #         The license call must stay inside apply_pysmile_license().
    #
    # Example structure only:
    #
    # pysmile.License((
    #     b"PASTE THE FIRST PART OF YOUR LICENSE HERE "
    #     b"PASTE THE NEXT PART OF YOUR LICENSE HERE "
    #     b"PASTE THE FINAL PART OF YOUR LICENSE HERE"
    # ))
    #
    # Do not commit the completed pysmile_license.py file to GitHub.
    raise RuntimeError(
        "PySMILE license is not configured. Copy pysmile_license.example.py "
        "to pysmile_license.py, delete this RuntimeError, and paste your own "
        "pysmile.License(...) call inside apply_pysmile_license()."
    )
