#!/bin/bash

# -m option: run library module as script
# --user makes pip install packages in home directory (no special privileges)

python3 -m pip install --user --upgrade pip
python3 -m pip --version
python --version

# Create a new virtual environment
python3 -m venv v_env

# Activate this environment’s packages/resources in isolation 
source v_env/bin/activate

# Inside bin/activate, there's a line that looks 
# like this: VIRTUAL_ENV="/Users/me/.envs/myenv"
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "Your virtual environment was created!"
    pip install -r requirements.txt
else
    echo "Your virtual environment is not working!"
fi

which python