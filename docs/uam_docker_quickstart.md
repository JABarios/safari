# SAFARI Docker Quickstart

This is the short version for collaborators who want to run SAFARI locally.

SAFARI runs in a local browser. EDF files stay on the local machine; they are
mounted into Docker and are not uploaded to an external server.

## Requirements

- Docker Desktop or Docker Engine.
- A folder with EDF/BDF files.
- A trained SAFARI model file named:

```text
safari_lgbm_v0.txt
```

## Folder Setup

Example:

```text
safari_project/
  data/
    rat_001.edf
    rat_002.edf
  models/
    safari_lgbm_v0.txt
  outputs/
```

## Build

From the SAFARI repository folder:

```bash
docker build -t safari .
```

## Run

Replace the paths with your local folders:

```bash
docker run --rm -p 8765:8765 \
  -v /absolute/path/to/data:/data:ro \
  -v /absolute/path/to/models:/models:ro \
  -v /absolute/path/to/outputs:/outputs \
  safari
```

Then open:

```text
http://127.0.0.1:8765
```

## What The Web App Does

The first version can:

- list EDF/BDF files in `/data`;
- run automatic Wake/NREM/REM staging;
- show state fractions and mean confidence;
- preview the first epochs;
- download CSV and NPZ outputs.

CSV columns:

```text
epoch,time_s,prediction,confidence,p_wake,p_nrem,p_rem
```

## Common Problems

### The web page says the model is missing

Check that this file exists:

```text
/absolute/path/to/models/safari_lgbm_v0.txt
```

Inside Docker it must appear as:

```text
/models/safari_lgbm_v0.txt
```

### No EDFs appear

Check that the mounted data folder contains `.edf` or `.bdf` files:

```bash
ls /absolute/path/to/data
```

### Port 8765 is busy

Use another local port:

```bash
docker run --rm -p 8766:8765 \
  -v /absolute/path/to/data:/data:ro \
  -v /absolute/path/to/models:/models:ro \
  -v /absolute/path/to/outputs:/outputs \
  safari
```

Then open:

```text
http://127.0.0.1:8766
```

## Research-Use Notice

SAFARI is an animal research tool. It is not intended for human clinical sleep
scoring or medical decision-making.

