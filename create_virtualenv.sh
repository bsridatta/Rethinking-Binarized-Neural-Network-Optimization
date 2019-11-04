# creating python virtualenv

set -e

virtualenv venv -p python3
source venv/bin/activate
pip3 --no-cache-dir install -r requirements.txt
pip3 install -e .
deactivate