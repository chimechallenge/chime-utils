import glob
import json
import logging
import os
import tarfile
from copy import deepcopy
from pathlib import Path
from typing import Optional

import soundfile as sf
from lhotse.recipes.chime6 import TimeFormatConverter
from lhotse.utils import Pathlike, resumable_download

from chime_utils.dgen.utils import tar_strip_members
from chime_utils.text_norm import get_txt_norm

CORPUS_URL = "https://us.openslr.org/resources/150/"
CHiME6_FS = 16000

# NOTE, CHiME-8 uses same original split as CHiME-6
chime7_map = {
    "train": [
        "S03",
        "S04",
        "S05",
        "S06",
        "S07",
        "S08",
        "S12",
        "S13",
        "S16",
        "S17",
        "S18",
        "S22",
        "S23",
        "S24",
    ],
    "dev": ["S02", "S09"],
    "eval": ["S19", "S20", "S01", "S21"],
}


def download_chime6(
    target_dir: Pathlike,
    force_download: Optional[bool] = False,
) -> Path:
    """
    Download and untar DiPCo dataset.
    :param target_dir: Pathlike, the path of the dir to storage the dataset.
    :param force_download: Bool, if True,
        download the tars no matter if the tars exist.
    :return: the path to downloaded and extracted directory with data.
    """

    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    for c_file in [
        "CHiME6_train.tar.gz",
        "CHiME6_dev.tar.gz",
        "CHiME6_eval.tar.gz",
        "CHiME6_transcriptions.tar.gz",
    ]:
        download_url = os.path.join(CORPUS_URL, c_file)
        tar_path = os.path.join(target_dir, c_file)
        resumable_download(
            download_url, filename=tar_path, force_download=force_download
        )
        if c_file.endswith(".tar.gz"):
            with tarfile.open(tar_path) as tar:
                strip = 2 if not c_file.endswith("transcriptions.tar.gz") else 1
                tar.extractall(
                    path=target_dir, members=tar_strip_members(target_dir, tar, strip)
                )

    return target_dir


def gen_chime6(
    output_dir, corpus_dir, download=False, dset_part="train,dev", challenge="chime8"
):
    """
    :param output_dir: Pathlike, path to output directory where the prepared data is saved.
    :param corpus_dir: Pathlike, path to the original CHiME-6 directory.
        If the dataset does not exist it will be downloaded
        to this folder if download is set to True.
    :param download: Whether to download the dataset from OpenSLR or not.
        You may have it already in storage.
    :param dset_part: Which part of the dataset you want to generate,
        choose between 'train','dev' and 'eval'.
        You can choose multiple ones by using commas e.g. 'train,dev,eval'.
    :param challenge: str, This option controls the text normalization used.
        Choose between 'chime7' and 'chime8'.
    """
    scoring_txt_normalization = get_txt_norm(challenge)
    corpus_dir = Path(corpus_dir).resolve()  # allow for relative path

    if download:
        download_chime6(corpus_dir)  # FIXME when openslr is ready

    def normalize_chime6(annotation, txt_normalizer):
        annotation_scoring = []
        for ex in annotation:
            ex["start_time"] = "{:.3f}".format(
                TimeFormatConverter.hms_to_seconds(ex["start_time"])
            )
            ex["end_time"] = "{:.3f}".format(
                TimeFormatConverter.hms_to_seconds(ex["end_time"])
            )
            if "ref" in ex.keys():
                del ex["ref"]
                del ex["location"]
                # cannot be used in inference
            ex_scoring = deepcopy(ex)
            ex_scoring["words"] = txt_normalizer(ex["words"])
            if len(ex_scoring["words"]) > 0:
                annotation_scoring.append(ex_scoring)
            # if empty remove segment from scoring
        return annotation, annotation_scoring

    splits = dset_part.split(",")
    # pre-create all destination folders
    for split in splits:
        Path(os.path.join(output_dir, "audio", split)).mkdir(
            parents=True, exist_ok=True
        )
        if split not in ["eval"]:
            Path(os.path.join(output_dir, "transcriptions", split)).mkdir(
                parents=True, exist_ok=True
            )
            Path(os.path.join(output_dir, "transcriptions_scoring", split)).mkdir(
                parents=True, exist_ok=True
            )
            Path(os.path.join(output_dir, "devices", split)).mkdir(
                parents=True, exist_ok=True
            )

        Path(os.path.join(output_dir, "uem", split)).mkdir(parents=True, exist_ok=True)

    all_uem = {k: [] for k in splits}
    for split in splits:
        json_dir = os.path.join(corpus_dir, "transcriptions", split)
        ann_json = glob.glob(os.path.join(json_dir, "*.json"))
        assert len(ann_json) > 0, (
            "CHiME-6 JSON annotation was not found in {}.\nPlease check if "
            "CHiME-6 data was downloaded correctly and the CHiME-6 main dir "
            "path is set correctly.\nYou can also download CHiME-6 using "
            "'--download' flag see '--help'.".format(json_dir)
        )
        # we also create audio files symlinks here
        audio_files = glob.glob(os.path.join(corpus_dir, "audio", split, "*.wav"))
        sess2audio = {}
        for x in audio_files:
            session_name = Path(x).stem.split("_")[0]
            if session_name not in sess2audio:
                sess2audio[session_name] = [x]
            else:
                sess2audio[session_name].append(x)

        # create device files
        for c_sess in sess2audio.keys():
            c_sess_audio_f = sess2audio[c_sess]
            devices_json = {}
            for audio in c_sess_audio_f:
                c_device = Path(audio).stem.lstrip(c_sess + "_")
                if c_device.startswith("P"):
                    # close talk device
                    d_type = {
                        "is_close_talk": True,
                        "speaker": c_device,
                        "channel": [1, 2],
                        "tot_channels": 2,
                        "device_type": "binaural_mic",
                    }
                else:
                    # array device
                    channel = c_device.split(".")[-1]
                    d_type = {
                        "is_close_talk": False,
                        "speaker": None,
                        "channel": channel,
                        "tot_channels": 4,
                        "device_type": "kinect_array",
                    }
                devices_json[c_device] = d_type

            if split not in ["eval"]:
                devices_json = dict(sorted(devices_json.items(), key=lambda x: x[0]))
                with open(
                    os.path.join(output_dir, "devices", split, c_sess + ".json"), "w"
                ) as f:
                    json.dump(devices_json, f, indent=4)

        # for each json file
        for j_file in ann_json:
            with open(j_file, "r") as f:
                annotation = json.load(f)
            sess_name = Path(j_file).stem

            annotation, scoring_annotation = normalize_chime6(
                annotation, scoring_txt_normalization
            )

            # create symlinks too
            for x in sess2audio[sess_name]:
                if Path(x).stem.split("_")[-1].startswith("P") and split in [
                    "eval",
                ]:
                    continue
                os.symlink(
                    x,
                    os.path.join(output_dir, "audio", split, Path(x).stem) + ".wav",
                )

            if split not in ["eval"]:
                with open(
                    os.path.join(
                        output_dir, "transcriptions", split, sess_name + ".json"
                    ),
                    "w",
                ) as f:
                    json.dump(annotation, f, indent=4)
                # retain original annotation but dump also the scoring one
                with open(
                    os.path.join(
                        output_dir,
                        "transcriptions_scoring",
                        split,
                        sess_name + ".json",
                    ),
                    "w",
                ) as f:
                    json.dump(scoring_annotation, f, indent=4)

            first = sorted([float(x["start_time"]) for x in annotation])[0]
            end = max([sf.SoundFile(x).frames for x in sess2audio[sess_name]])
            c_uem = "{} 1 {} {}\n".format(
                sess_name,
                "{:.3f}".format(float(first)),
                "{:.3f}".format(end / CHiME6_FS),
            )
            all_uem[split].append(c_uem)

    for k in all_uem.keys():
        c_uem = all_uem[k]
        if len(c_uem) > 0:
            c_uem = sorted(c_uem)
            with open(os.path.join(output_dir, "uem", k, "all.uem"), "w") as f:
                f.writelines(c_uem)

        logging.info(f"CHiME-6 {k} set generated successfully.")
