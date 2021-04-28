#!/bin/bash

# -m option: run library module as script
# pip defaults to installing Python packages to a 
# system directory (such as /usr/local/lib/python3.4). 
# This requires root access.
# --user makes pip install packages in your home directory
# instead, which doesn't require any special privileges.
# install pip manually to ensure having the latest version
# it's recommended to uses the system pip to bootstrap
# a user installation
# https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/
python3 -m pip install --user --upgrade pip
python3 -m pip --version
python --version

# Create a new virtual environment
python3 -m venv v_env

# Activate this environment’s packages/resources in isolation 
source v_env/bin/activate

# Inside the script at bin/activate, there's a line that looks 
# like this: VIRTUAL_ENV="/Users/me/.envs/myenv"
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "Your virtual environment was created!"
    pip install -r requirements.txt
else
    echo "Your virtual environment is not working!"
fi

which python