# Data Preprocessing
This folder contains files necessary for the indexing process.
Before one can process our data for training, they need to convert any corpus into plain text and arrange the data as in [data](https://github.com/Victor-wang-902/SUREALM/tree/main/data). [process_raw.py](https://github.com/Victor-wang-902/SUREALM/blob/main/src/preprocessing/process_raw.py) provides example script for converting [DSTC9 track 1](https://github.com/alexa/alexa-with-dstc9-track1-dataset) dataset.
Also note that in the data config, the field `dropoff` for each file is optional and can be omitted.