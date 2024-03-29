# CHiME Utils

[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/chimechallenge.svg?style=social&label=Follow%20%40chimechallenge)](https://twitter.com/chimechallenge)
[![Slack][slack-badge]][slack-invite]

✅ Official data generation and data preparation scripts for [CHiME-8 DASR](https://www.chimechallenge.org/current/task1/index). 

We provide a more convenient standalone interface for downloading and prepare the core CHiME-8 DASR data. <br> 
This year we also support automatic downloading of CHiME-6. <br>

---

⚠️ **NOTE** <br>
For in-depth details about CHiME-8 DASR data and rules refer to [chimechallenge.org/current/task1/data](https://www.chimechallenge.org/current/task1/data).  

<h4>📩 Contact </h4>
For any issue/bug/question with this package feel free to raise an issue here or reach us via the [CHiME Slack](https://join.slack.com/t/chime-fey5388/shared_invite/zt-1oha0gedv-JEUr1mSztR7~iK9AxM4HOA).

---

## Installation

You can install with: 

`pip install git+https://github.com/chimechallenge/chime-utils`


## Usage 

This package brings a new module: <br>
`from chime-utils import dgen, dprep, scoring, text_norm`

And new CLI commands: <br>
- `chime-utils dgen` <br>
    - generates and downloads CHiME-8 data.
- `chime-utils lhotse-prep` <br>
    - prepares CHiME-8 data lhotse manifests.
- `chime-utils espnet-prep` <br>
    - prepares CHiME-8 data ESPNet/Kaldi-style manifests.
- `chime-utils score` <br>
    - scripts used for official scoring.

Hereafter we describe each command/function in detail. 

## Data generation

### ⚡ All DASR data in one go (recommended)

You can generate all CHiME-8 DASR data in one go with: <br>
`chime-utils dgen dasr ./download /path/to/mixer6_root ./chime8_dasr --part train,dev --download` <br>
If unsure you can always use some `--help`.

This script will download CHiME-6, DiPCo and NOTSOFAR1 automatically in `./download` <br>
Ensure you have at least 1TB of space there. You can remove the `.tar.gz` after the full data preparation to save some space later.

Mixer 6 Speech instead has to be obtained through LDC. <br>
Refer to [chimechallenge.org/current/task1/data](https://www.chimechallenge.org/current/task1/data) on how to obtain Mixer 6 Speech. <br>
The Mixer 6 root folder should look like this: <br>

```
mixer6_root 
├── data 
│   └── pcm_flac 
├── metadata 
│   ├── iv_components_final.csv 
│   ├── mx6_calls.csv 
...
├── splits 
│   ├── dev_a 
│   ├── dev_a.list 
...
└── train_and_dev_files
```

🔐 You can check if the data has been successfully prepared with: <br>
`chime-utils dgen checksum ./chime8_dasr` <br>
The resulting `./chime8_dasr` should look like this: 

```
chime8_dasr/
├── chime6/
├── dipco/
├── mixer6/
└── notsofar1/
```

For more information about CHiME-8 data you should look at the website data page: [chimechallenge.org/current/task1/data](https://www.chimechallenge.org/current/task1/data).

### 🐢 Single Dataset Scripts

We provide scripts for obtaining each core dataset independently if needed. <br>
Command basic usage: `chime-utils dgen <DATASET> <DOWNLOAD_DIR> <OUTPUT_DIR> --download` <br>


- CHiME-6 (not CHiME-5 !)
   - `chime-utils dgen chime6 ./download/chime6 ./chime8_dasr/chime6 --part train,dev --download` 
     - If it is already in storage in `/path/to/chime6_root` instead you can use:
       - `chime-utils dgen chime6 /path/to/chime6_root ./chime8_dasr/chime6 --part train,dev`
- DiPCo
   - `chime-utils dgen dipco ./download/dipco ./chime8_dasr/dipco --part train,dev --download` 
     - If it is already in storage in `/path/to/dipco` instead you can use:
       - `chime-utils dgen dipco /path/to/dipco ./chime8_dasr/dipco --part train,dev`
- Mixer 6 Speech
    - `chime-utils dgen mixer6 /path/to/mixer6_root ./chime8_dasr/mixer6 --part train_call,train_intv,train,dev`
      - It must be obtained via LDC ([again, see CHiME-8 data page](https://www.chimechallenge.org/current/task1/data)) and extracted manually. 
- NOTSOFAR1
   - `chime-utils dgen notsofar1 ./download/notsofar1 ./chime8_dasr/notsofar1 --part train,dev --download` 
     - If it is already in storage in `/path/to/notsofar1` instead you can use:
       - `chime-utils dgen notsofar1 /path/to/notsofar1 ./chime8_dasr/notsofar1 --part train,dev`

NOTSOFAR download folder should look like this: 

```
.
├── dev
│   └── dev_set
│       └── 240208.2_dev
└── train
    └── train_set
        └── 240208.2_train
```


## 🚀 NVIDIA NeMo Official Baseline 
 
[![nVIDIA](https://img.shields.io/badge/nVIDIA-%2376B900.svg?style=for-the-badge&logo=nVIDIA&logoColor=white)](https://github.com/NVIDIA/NeMo)

This year CHiME-8 DASR baseline is built directly upon NVIDIA NeMo team last year [CHiME-7 DASR Submission](https://arxiv.org/pdf/2310.12378.pdf) [1]. <br>

It is available at [chimechallenge/C8DASR-Baseline-NeMo/tree/main/scripts/chime8](https://github.com/chimechallenge/C8DASR-Baseline-NeMo/tree/main/scripts/chime8), we used our own fork of NeMo because it will be easier to maintain during the challenge.



## Data preparation

For convenience, we also offer here data preparation scripts for different toolkits (parsing the data into different manifests formats):
- [Kaldi](https://github.com/kaldi-asr/kaldi) and [K2/Icefall](https://github.com/k2-fsa/icefall) (with [lhotse](https://github.com/lhotse-speech/lhotse))
- [ESPNet](https://github.com/espnet/espne) (with [lhotse](https://github.com/lhotse-speech/lhotse))

See [DATAPREP.md](./DATAPREP.md). 


## Scoring

Last but not least, we also provide scripts for scoring (the exact same scripts organizers will use for ranking CHiME-8 DASR submissions). <br>
To learn more about scoring and ranking in CHiME-8 DASR please head over the official [CHiME-8 Challenge website](https://www.chimechallenge.org/current/task1/index).

Note that the following scrips expect the participants predictions to be in the standard CHiME-style JSON format also known as **SegLST (Segment-wise Long-form Speech Transcription) format** (we adopt [Meeteval](https://github.com/fgnt/meeteval) naming convention [2]).
<br>
Each SegLST is a JSON containing a list of 
dicts (one for each utterance) with the following keys:

```
    {
        "end_time": "43.82",
        "start_time": "40.60",
        "words": "chime style json format",
        "speaker": "P05",
        "session_id": "S02"
    }
```

Please head over to [CHiME-8 DASR Submission instructions](https://www.chimechallenge.org/current/task1/submission) to know more about scoring and text normalization.

The scripts may accept a single SegLST JSON or a folder where multiple SegLST JSON files are contained. <br>
E.g. one per each scenario as requested in [CHiME-8 DASR Submission instructions](https://www.chimechallenge.org/current/task1/submission). <br>
For example for the development set: <br>

```
dev
├── chime6.json
├── dipco.json
├── mixer6.json
└── notsofar1.json
```

### CHiME-8 DASR Ranking Score


### Text Normalization


Text normalization is applied automatically before scoring to your predictions. <br>
In CHiME-8 DASR we use a more complex text normalization which is built on top of Whisper text normalization but is crucially different (less "aggressive" and made idempotent). <br>
Examples are available here: [./tests/test_normalizer.py](./tests/test_normalizer.py)


### ASR 

In detail, we provide scripts to compute common ASR metrics for long-form meeting scenarios. 
These scores are computed through the awesome [Meeteval](https://github.com/fgnt/meeteval) [2] toolkit. 

Command basic usage: `chime-utils score <SCORE_TYPE> -s <PREDICTIONS_DIR> <CHIME-8 ROOT> --dset-part <DEV or EVAL> --ignore-missing` <br>
`--ignore-missing` will ignore from scoring the scenarios which do not have for that `--dset-part` system predictions or ground truth (e.g. NOTSOFAR1 dev).

We expect the predictions folder to be in the format outlined previously (dev folder and then JSON SegLST for each scenario). <br>
Two metrics are supported:

- tcpWER [2] `chime-utils score tcpwer`
- concatenated minimum-permutation word error rate (cpWER) [3] `chime-utils score cpwer`

You can also use `chime-utils score segslt2ctm input-dir output-dir` to automatically convert all SegLST JSON files in `input-dir` and its subfolders to `.ctm` files. <br> 
This allows to use easily also other ASR metrics tools such as [NIST Asclite](https://mig.nist.gov/MIG_Website/tools/asclite.html). 


### Error Analysis

As well as utils to convert SegSLT (aka CHiME-6 style) JSON annotation to other formats such as .ctm and Audacity compatible labels (.txt)
so that systems output can be more in-depth analyzed. 

- [Segment Time Marked](https://www.nist.gov/system/files/documents/2021/08/03/OpenASR20_EvalPlan_v1_5.pdf) `.stm` format conversion:
   - `chime-utils score segslt2stm input-dir output-dir`
- [Conversation Time Mark](https://web.archive.org/web/20170119114252/http://www.itl.nist.gov/iad/mig/tests/rt/2009/docs/rt09-meeting-eval-plan-v2.pdf) `.ctm` format conversion:
   - `chime-utils score segslt2ctm input-dir output-dir`
- [Rich Transcription Time Marked](https://web.archive.org/web/20170119114252/http://www.itl.nist.gov/iad/mig/tests/rt/2009/docs/rt09-meeting-eval-plan-v2.pdf) `.rttm` format conversion:
   - `chime-utils score segslt2rttm input-dir output-dir`
       - this allows to use other diarization scoring tools such as [dscore](https://github.com/nryant/dscore).


#### 🔍 MeetEval meeting recognition visualization (recommended)

For ASR+diarization error analysis we recommend the use of this super useful [Meeteval](github.com/fgnt/meeteval) tool (will be presented at ICASSP 2024 in a show and tell session): <br>
- [https://fgnt.github.io/meeteval_jupyterlite/lab](https://fgnt.github.io/meeteval_jupyterlite/lab?path=Demo.ipynb)

You can upload your predictions and references in any supported file format (SegLST is preferred) to the [JupyterLite notebook running in the browser](https://fgnt.github.io/meeteval_jupyterlite/lab?path=Demo.ipynb), run the visualization in a Jupyter notebook locally, or build a visualization in the command-line using
```
python -m meeteval.viz html -r ref.stm -h mysystem:hyp.stm -h myothersystem:hyp.stm --out=out --regex='.*'
```
which will generate HTML files that you can open with any browser.

---


## Contribute

If you wish to contribute, download this repo:

`git clone https://github.com/chimechallenge/chime-utils` <br>
`cd chime-utils` <br>

and then install with:  

`pip install -e .` <br>
`pip install pre-commit` <br>
`pre-commit install --install-hooks`

---

## References 

[1] Park, T. J., Huang, H., Jukic, A., Dhawan, K., Puvvada, K. C., Koluguri, N., ... & Ginsburg, B. (2023). The CHiME-7 Challenge: System Description and Performance of NeMo Team's DASR System. arXiv preprint arXiv:2310.12378. <br>

[2] von Neumann, T., Boeddeker, C., Delcroix, M., & Haeb-Umbach, R. (2023). MeetEval: A Toolkit for Computation of Word Error Rates for Meeting Transcription Systems. arXiv preprint arXiv:2307.11394. <br>

[3] Watanabe, S., Mandel, M., Barker, J., Vincent, E., Arora, A., Chang, X., ... & Ryant, N. (2020). CHiME-6 challenge: Tackling multispeaker speech recognition for unsegmented recordings. arXiv preprint arXiv:2004.09249.

[4] Cornell, S., Wiesner, M., Watanabe, S., Raj, D., Chang, X., Garcia, P., ... & Khudanpur, S. (2023). The CHiME-7 DASR Challenge: Distant Meeting Transcription with Multiple Devices in Diverse Scenarios. arXiv preprint arXiv:2306.13734.

[slack-badge]: https://img.shields.io/badge/slack-chat-green.svg?logo=slack
[slack-invite]: https://join.slack.com/t/chime-fey5388/shared_invite/zt-1oha0gedv-JEUr1mSztR7~iK9AxM4HOA
[twitter]: https://twitter.com/chimechallenge<h2>References</h2>
