# Python Virtual Environment

Most dependencies are installed via MacPorts. Only `vl-convert-python` (unavailable
in MacPorts) goes into the venv, which inherits MacPorts packages via
`--system-site-packages`.

## Create and populate

```
# MacPorts packages
sudo port install py314-pandas py314-numpy py314-requests py314-joblib py314-altair py314-lxml

# Slim venv for vl-convert-python only
python3 -m venv --system-site-packages .venv
.venv/bin/pip install vl-convert-python
```

## Run

```
.venv/bin/python nvperf.py
```

Or activate for the session (then use `python` and `./nvperf.py` directly):

```
source .venv/bin/activate
./nvperf.py
deactivate
```

## Update packages

```
# MacPorts packages
sudo port upgrade py314-pandas py314-numpy py314-requests py314-joblib py314-altair py314-lxml

# venv-only package
.venv/bin/pip install --upgrade vl-convert-python
```

## Delete

```
rm -rf .venv
```
