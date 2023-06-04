## Running Locally

1. Clone the repository: `git clone git@github.com:e-cal/nn-from-scratch.git`
2. (Optional) Create a virtual environment
   ```bash
   venv "$VIRTUALENV_HOME/ENVNAME"
   source "$VIRTUALENV_HOME/ENVNAME/bin/activate"
   ```
   See the next section for creating a notebook kernel in the virtual environment.
3. Install the requirements: `pip install -r requirements.txt`
4. Launch the notebook server: `jupyter notebook`
5. Select the correct kernel to use for the notebook:
   Kernel > Change kernel > [select the proper kernel]

### Setting up a Notebook Kernel in a Virtual Environment

If you install the required packages to a virtual environment, you can add the
environment to Jupyter with this command: `python3 -m ipykernel install --name=ENVNAME`
(make sure to replace ENVNAME. You may also need to add the `--user` flag if you get permission errors.)

Should print something like:

```
Installed kernelspec ENVNAME in /home/USER/.local/share/jupyter/kernels/ENVNAME
```

> If there is any difficulty getting a kernel set up with the required packages
> installed, you can install the requirements outside of a virtual environment
> and use the default kernel.
