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

import soundfile as sf
from lhotse.utils import Pathlike, resumable_download

from chime_utils.dgen.utils import get_mappings, tar_strip_members
from chime_utils.text_norm import get_txt_norm

logging.basicConfig(
    format=(
        "%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d]" " %(message)s"
    ),
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

CORPUS_URL = "https://zenodo.org/records/8122551/files/DipCo.tgz"
DIPCO_FS = 16000

dipco_c8_sess2split = {
    "S01": "eval",
    "S02": "train",
    "S03": "eval",
    "S04": "dev",
    "S05": "dev",
    "S06": "eval",
    "S07": "eval",
    "S08": "eval",
    "S09": "train",
    "S10": "train",
}


def download_dipco(
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
    tar_path = os.path.join(target_dir, "DiPCo.tgz")
    resumable_download(CORPUS_URL, filename=tar_path, force_download=force_download)
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=target_dir, members=tar_strip_members(target_dir, tar, 1))

    return target_dir


def gen_dipco(
    output_dir, corpus_dir, download=False, dset_part="train,dev", challenge="chime8"
):
    """
    :param output_dir: Pathlike,
        the path of the dir to storage the final dataset
        (note that we will use symbolic links to the original dataset where
        possible to minimize storage requirements).
    :param corpus_dir: Pathlike, the original path to DiPCo root folder.
    :param download: bool, whether to download the dataset or not (you may have
        it already in storage).
    :param dset_part: str, choose between 'train',
        'dev' and 'eval' or 'train,dev' for multiple.
    :param challenge: str, choose between chime7 and chime8, it controls the
        choice of the text normalization and possibly how sessions are split
        between dev and eval.
    """
    corpus_dir = Path(corpus_dir).resolve()  # allow for relative path
    mapping = get_mappings(challenge)
    spk_map = mapping["spk_map"]["dipco"]
    sess_map = mapping["sessions_map"]["dipco"]
    text_normalization = get_txt_norm(challenge)

    assert challenge == "chime8"  # not implemented chime7 currently

    if download:
        download_dipco(corpus_dir)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    def normalize_dipco(annotation, txt_normalizer, split):
        annotation_scoring = []

        def _get_time(x):
            return (dt.strptime(x, "%H:%M:%S.%f") - dt(1900, 1, 1)).total_seconds()

        for indx in range(len(annotation)):
            ex = annotation[indx]
            ex["session_id"] = sess_map[ex["session_id"]]
            ex["start_time"] = "{:.3f}".format(_get_time(ex["start_time"]["U01"]))
            ex["end_time"] = "{:.3f}".format(_get_time(ex["end_time"]["U01"]))
            ex["speaker"] = spk_map[ex["speaker_id"]]
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
    for dest_split in dset_part:
        assert dest_split in ["train", "dev", "eval"]
        # here same splits no need to remap
        Path(os.path.join(output_dir, "audio", dest_split)).mkdir(
            parents=True, exist_ok=True
        )
        if dest_split not in ["eval", "dev"]:
            Path(os.path.join(output_dir, "transcriptions", dest_split)).mkdir(
                parents=True, exist_ok=True
            )
            Path(os.path.join(output_dir, "transcriptions_scoring", dest_split)).mkdir(
                parents=True, exist_ok=True
            )

            Path(os.path.join(output_dir, "devices", dest_split)).mkdir(
                parents=True, exist_ok=True
            )

        # now we fetch all possible json files here, including evaluation ones
        ann_json = []
        audio_files = []
        for orig_split in ["dev", "eval"]:
            json_dir = os.path.join(corpus_dir, "transcriptions", orig_split)
            ann_json.extend(glob.glob(os.path.join(json_dir, "*.json")))
            audio_files.extend(
                glob.glob(os.path.join(corpus_dir, "audio", orig_split, "*.wav"))
            )
        assert len(ann_json) > 0, (
            "DiPCo JSON annotation was not found in {}.\nPlease check if "
            "DiPCo data was downloaded correctly and the DiPCo main dir "
            "path is set correctly.\n"
            "You can also download DiPCo using '--download' flag see '--help'.".format(
                json_dir
            )
        )

        # we also create audio files symlinks here
        sess2audio = {}
        for x in audio_files:
            session_name = Path(x).stem.split("_")[0]
            # check now if this session is in the current destination split
            # if dest_split in ["eval", "dev"] and Path(x).stem.split("_")[-1].startswith("P"):
            # you should not use close-talk in evaluation !
            #    continue
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
            if dipco_c8_sess2split[sess_name] != dest_split:
                continue

            annotation, scoring_annotation = normalize_dipco(
                annotation, text_normalization, dest_split
            )

            annotation = sorted(annotation, key=lambda x: float(x["start_time"]))
            scoring_annotation = sorted(
                scoring_annotation, key=lambda x: float(x["start_time"])
            )

            new_sess_name = sess_map[sess_name]
            # create symlinks too but swap names for the sessions too
            devices_info = {}
            for x in sess2audio[sess_name]:
                filename = new_sess_name + "_" + "_".join(Path(x).stem.split("_")[1:])
                if filename.split("_")[1].startswith("P"):
                    speaker_id = filename.split("_")[1]
                    filename = filename.split("_")[0] + "_{}".format(
                        spk_map[speaker_id]
                    )

                    devices_info[filename] = {
                        "is_close_talk": True,
                        "speaker": spk_map[speaker_id],
                        "num_channels": 1,
                        "device_type": "headset_mic",
                    }
                else:
                    channel = Path(x).stem.split(".")[-1]
                    devices_info[Path(x).stem.lstrip(sess_name)] = {
                        "is_close_talk": False,
                        "speaker": None,
                        "num_channels": 1,
                        "device_type": f"circular_array_{channel}_mic",
                    }

                if not (
                    dest_split in ["eval", "dev"]
                    and Path(x).stem.split("_")[-1].startswith("P")
                ):
                    os.symlink(
                        x,
                        os.path.join(
                            output_dir, "audio", dest_split, filename + ".wav"
                        ),
                    )

            devices_info = dict(sorted(devices_info.items(), key=lambda x: x[0]))

            if dest_split not in ["dev", "eval"]:
                with open(
                    os.path.join(
                        output_dir, "devices", dest_split, sess_map[sess_name] + ".json"
                    ),
                    "w",
                ) as f:
                    json.dump(devices_info, f, indent=4)

                with open(
                    os.path.join(
                        output_dir,
                        "transcriptions",
                        dest_split,
                        new_sess_name + ".json",
                    ),
                    "w",
                ) as f:
                    json.dump(annotation, f, indent=4)
                with open(
                    os.path.join(
                        output_dir,
                        "transcriptions_scoring",
                        dest_split,
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
            Path(os.path.join(output_dir, "uem", dest_split)).mkdir(parents=True)
            to_uem = sorted(to_uem)
            with open(os.path.join(output_dir, "uem", dest_split, "all.uem"), "w") as f:
                f.writelines(to_uem)
