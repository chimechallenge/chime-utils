### Data Preparation Scripts for Different Toolkits 


⚠️ **NOTE** <br>
In all manifests preparation scripts you can choose which text normalization you 
want to apply on each utterance using as an additional argument `--txt-norm`.<br>
You can choose between `chime8`, `chime7`, `chime6` and `none` for no text normalization.


### K2/Icefall/Lhotse

You can prepare Lhotse manifests compatible with [K2/Icefall](https://github.com/k2-fsa/icefall) for all core datasets easily.

For example, for CHiME-6:
- e.g. to prepare manifests for far-field arrays and training, development partition: 
    - `chime-utils lhotse-prep chime6 ./chime8_dasr/chime6 ./manifests/lhotse/chime6 --dset-part train,dev --mic mdm`
- you can also prepare manifests for on speakers close-talk mics: 
    - `chime-utils lhotse-prep chime6 ./chime8_dasr/chime6 ./manifests/lhotse/chime6 --dset-part train,dev --mic ihm`

Similarly, you can use `chime-utils lhotse-prep dipco`, `chime-utils lhotse-prep mixer6` and `chime-utils lhotse-prep notsofar1`
commands to prepare manifests for the other three scenarios. 


### ESPNet and Kaldi

You can prepare Kaldi and ESPNet manifests for all core datasets easily. <br>

For example, for CHiME-6:
- e.g. to prepare manifests for far-field arrays and training, development partition: 
    - `chime-utils espnet-prep chime6 ./chime8_dasr/chime6 ./manifests/espnet/chime6 --dset-part train,dev --mic mdm`
- you can also prepare manifests for on speakers close-talk mics: 
    - `chime-utils espnet-prep chime6 ./chime8_dasr/chime6 ./manifests/espnet/chime6 --dset-part train,dev --mic ihm`

Similarly, you can use `chime-utils espnet-prep dipco`, `chime-utils espnet-prep mixer6` and `chime-utils espnet-prep notsofar1`
commands to prepare manifests for the other three scenarios. 
