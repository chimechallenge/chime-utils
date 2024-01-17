import glob
import json
import logging
import os
import os.path
import tarfile
from copy import deepcopy
from datetime import datetime as dt
from pathlib import Path
from typing import Optional
from chime8_macro import dipco_map, dipco_spk_offset
from chime_utils.text_norm import get_txt_norm

import soundfile as sf
from lhotse.utils import Pathlike, resumable_download, safe_extract

logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

CORPUS_URL = "https://zenodo.org/records/8122551/files/DipCo.tgz"
DIPCO_FS = 16000


def download_dipco(
    target_dir: Pathlike,
    force_download: Optional[bool] = False,
) -> Path:
    """
    Download and untar DiPCo dataset.
    :param target_dir: Pathlike, the path of the dir to storage the dataset.
    :param force_download: Bool, if True, download the tars no matter if the tars exist.
    :return: the path to downloaded and extracted directory with data.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    tar_path = os.path.join(target_dir, "DiPCo.tgz")
    resumable_download(CORPUS_URL, filename=tar_path, force_download=force_download)
    with tarfile.open(tar_path) as tar:
        safe_extract(tar, path=target_dir)

    return target_dir


def gen_dipco(
    output_dir, corpus_dir, download=False, dset_part="dev", challenge="chime8"
):
    """
    :param output_dir: Pathlike, the path of the dir to storage the final dataset
        (note that we will use symbolic links to the original dataset where
        possible to minimize storage requirements).
    :param corpus_dir: Pathlike, the original path to DiPCo root folder.
    :param download: bool, whether to download the dataset or not (you may have
        it already in storage).
    :param dset_part: str, choose between 'dev' and 'eval' or 'dev,eval' for both.
    :param challenge: str, choose between chime7 and chime8, it controls the
        choice of the text normalization and possibly how sessions are split
        between dev and eval.
    """
    assert dset_part in ["dev", "eval"]

    text_normalization = get_txt_norm(challenge)

    if download:
        download_dipco(corpus_dir)
        # need this because it will be extracted in a subfolder
        corpus_dir = os.path.join(corpus_dir, "Dipco")

    if os.path.exists(output_dir):
        logger.warning("The output directory {} already exists. "
                       "This may possible overwrite previous data.".format(output_dir))
    else:
        Path(output_dir).mkdir(parents=True, exist_ok=False)

    def normalize_dipco(annotation, txt_normalizer, split):
        annotation_scoring = []

        def _get_time(x):
            return (dt.strptime(x, "%H:%M:%S.%f") - dt(1900, 1, 1)).total_seconds()

        for indx in range(len(annotation)):
            ex = annotation[indx]
            ex["session_id"] = dipco_map[split][ex["session_id"]]

            ex["start_time"] = "{:.3f}".format(_get_time(ex["start_time"]["U01"]))
            ex["end_time"] = "{:.3f}".format(_get_time(ex["end_time"]["U01"]))
            ex["speaker"] = "P{:02d}".format(
                dipco_spk_offset + int(ex["speaker_id"].strip("P"))
            )
            del ex["speaker_id"]

            new_ex = {}
            for k in ex.keys():
                if k in [
                    "speaker",
                    "start_time",
                    "end_time",
                    "words",
                    "session_id",
                ]:
                    new_ex[k] = ex[k]
            annotation[indx] = new_ex
            ex = annotation[indx]
            # cannot be used in inference
            ex_scoring = deepcopy(ex)
            ex_scoring["words"] = txt_normalizer(ex["words"])
            if ex_scoring["words"]:
                annotation_scoring.append(ex_scoring)
            # if empty remove segment from scoring

        return annotation, annotation_scoring

    dset_part = dset_part.split(",")
    for split in dset_part:
        # here same splits no need to remap
        Path(os.path.join(output_dir, "audio", split)).mkdir(
            parents=True, exist_ok=True
        )
        Path(os.path.join(output_dir, "transcriptions", split)).mkdir(
            parents=True, exist_ok=True
        )
        Path(os.path.join(output_dir, "transcriptions_scoring", split)).mkdir(
            parents=True, exist_ok=True
        )
        json_dir = os.path.join(corpus_dir, "transcriptions", split)
        ann_json = glob.glob(os.path.join(json_dir, "*.json"))
        assert len(ann_json) > 0, (
            "DiPCo JSON annotation was not found in {}, please check if "
            "DiPCo data was downloaded correctly and the DiPCo main dir "
            "path is set correctly".format(json_dir)
        )

        # we also create audio files symlinks here
        audio_files = glob.glob(os.path.join(corpus_dir, "audio", split, "*.wav"))
        sess2audio = {}
        for x in audio_files:
            session_name = Path(x).stem.split("_")[0]
            if split == "eval" and Path(x).stem.split("_")[-1].startswith("P"):
                # you should not use close-talk in evaluation !
                continue
            if session_name not in sess2audio:
                sess2audio[session_name] = [x]
            else:
                sess2audio[session_name].append(x)

        # for each json file
        to_uem = []
        for j_file in ann_json:
            with open(j_file, "r") as f:
                annotation = json.load(f)
            sess_name = Path(j_file).stem

            annotation, scoring_annotation = normalize_dipco(
                annotation, text_normalization, split
            )

            annotation = sorted(annotation, key=lambda x: float(x["start_time"]))
            scoring_annotation = sorted(
                scoring_annotation, key=lambda x: float(x["start_time"])
            )

            new_sess_name = dipco_map[split][sess_name]
            # create symlinks too but swap names for the sessions too
            for x in sess2audio[sess_name]:
                filename = new_sess_name + "_" + "_".join(Path(x).stem.split("_")[1:])
                if filename.split("_")[1].startswith("P"):
                    speaker_id = filename.split("_")[1]
                    filename = filename.split("_")[0] + "_P{:02d}".format(
                        dipco_spk_offset + int(speaker_id.strip("P"))
                    )
                os.symlink(
                    x, os.path.join(output_dir, "audio", split, filename + ".wav")
                )

            with open(
                os.path.join(
                    output_dir, "transcriptions", split, new_sess_name + ".json"
                ),
                "w",
            ) as f:
                json.dump(annotation, f, indent=4)
            with open(
                os.path.join(
                    output_dir,
                    "transcriptions_scoring",
                    split,
                    new_sess_name + ".json",
                ),
                "w",
            ) as f:
                json.dump(scoring_annotation, f, indent=4)

            uem_start = 0
            uem_end = max([sf.SoundFile(x).frames for x in sess2audio[sess_name]])
            c_uem = "{} 1 {} {}\n".format(
                new_sess_name,
                "{:.3f}".format(float(uem_start)),
                "{:.3f}".format(float(uem_end / DIPCO_FS)),
            )
            to_uem.append(c_uem)

        if len(to_uem) > 0:
            assert split in ["dev", "eval"]  # uem only for development set
            Path(os.path.join(output_dir, "uem", split)).mkdir(parents=True)
            to_uem = sorted(to_uem)
            with open(os.path.join(output_dir, "uem", split, "all.uem"), "w") as f:
                f.writelines(to_uem)
