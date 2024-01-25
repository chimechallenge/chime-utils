import glob
import json
import os
from copy import deepcopy
from pathlib import Path

import soundfile as sf

from chime_utils.dgen.utils import get_mappings
from chime_utils.text_norm import get_txt_norm

devices2type = {
    "CH01": "lavaliere",
    "CH02": "headmic",
    "CH03": "lavaliere",
    "CH04": "podium_mic",
    "CH05": "PZM_mic",
    "CH06": "AT3035_Studio_mic",
    "CH07": "ATPro45_Hanging_mic",
    "CH08": "Panasonic_Camcorder",
    "CH09": "RODE_NT6_mic",
    "CH10": "RODE_NT6_mic",
    "CH11": "Samson_C01U_mic",
    "CH12": "AT815b_Shotgun_mic",
    "CH13": "Acoustimagic_array",
}


def read_list_file(list_f):
    with open(list_f, "r") as f:
        lines = f.readlines()
    out = {}
    for l in lines:  # noqa E741
        c_line = l.rstrip("\n").split("\t")
        session_id = c_line[0]
        subject_id, intv_id = c_line[1].split(",")
        out[session_id] = [subject_id, intv_id]
    return out


def gen_mixer6(
    output_dir,
    corpus_dir,
    dset_part="train_call,train_intv,dev",
    challenge="chime8",
):
    """
    :param output_dir: Pathlike,
        the path of the dir to storage the final dataset
        (note that we will use symbolic links to the original dataset where
        possible to minimize storage requirements).
    :param corpus_dir: Pathlike,
        the original path to Mixer 6 Speech root folder.
    :param download: bool, whether to download the dataset or not (you may have
        it already in storage).
    :param dset_part: str, choose between
    'train_intv', 'train_call','dev' and 'eval' or
    'train_intv,train_call' for both.
    :param challenge: str, choose between chime7 and chime8, it controls the
        choice of the text normalization and possibly how sessions are split
        between dev and eval.
    """
    corpus_dir = Path(corpus_dir).resolve()  # allow for relative path
    mapping = get_mappings(challenge)
    spk_map = mapping["spk_map"]["mixer6"]
    sess_map = mapping["sessions_map"]["mixer6"]
    scoring_txt_normalization = get_txt_norm(challenge)

    def normalize_mixer6(annotation, txt_normalizer):
        annotation_scoring = []
        for indx in range(len(annotation)):
            ex = annotation[indx]
            ex_scoring = deepcopy(ex)
            ex_scoring["words"] = txt_normalizer(ex["words"])
            if len(ex_scoring["words"]) > 0:
                annotation_scoring.append(ex_scoring)
            # if empty remove segment from scoring
        return annotation, annotation_scoring

    def create_audio_symlinks(
        split,
        tgt_sess_name,
        audios,
        output_dir,
        interviewer_name,
        subject_name,
    ):
        # we also create a JSON that describes each device
        devices_json = {}
        for c_audio in audios:
            audioname = Path(c_audio).stem
            channel_num = int(audioname.split("_")[-1].strip("CH"))
            if channel_num <= 3 and split == "eval":
                continue
            new_name = "{}_CH{:02d}".format(tgt_sess_name, channel_num)
            os.symlink(
                c_audio,
                os.path.join(output_dir, "audio", split, new_name + ".flac"),
            )
            if channel_num <= 3:
                c_spk_mic_name = (
                    interviewer_name if channel_num in [1, 3] else subject_name
                )
                devices_json["CH{}".format(channel_num)] = {
                    "is_close_talk": True,
                    "speaker": c_spk_mic_name,
                    "num_channels": 1,
                    "device_type": devices2type["CH{:02d}".format(channel_num)],
                }
            else:
                devices_json["CH{}".format(channel_num)] = {
                    "is_close_talk": False,
                    "speaker": None,
                    "num_channels": 1,
                    "device_type": devices2type["CH{:02d}".format(channel_num)],
                }

        out_json = os.path.join(output_dir, "devices", split, f"{tgt_sess_name}.json")
        Path(out_json).parent.mkdir(exist_ok=True, parents=True)
        devices_json = dict(
            sorted(devices_json.items(), key=lambda x: int(x[0].strip("CH")))
        )
        with open(out_json, "w") as f:
            json.dump(devices_json, f, indent=4)

    splits = dset_part.split(",")
    audio_files = glob.glob(
        os.path.join(
            corpus_dir,
            "data/pcm_flac",
            "**/*.flac",
        ),
        recursive=True,
    )
    sess2audio = {}

    for x in audio_files:
        session_name = "_".join(Path(x).stem.split("_")[0:-1])
        if session_name not in sess2audio:
            sess2audio[session_name] = [x]
        else:
            sess2audio[session_name].append(x)

    for c_split in splits:
        assert c_split in ["train_intv", "train_call", "dev", "eval"]
        Path(os.path.join(output_dir, "audio", c_split)).mkdir(
            parents=True, exist_ok=False
        )
        Path(os.path.join(output_dir, "transcriptions", c_split)).mkdir(
            parents=True, exist_ok=False
        )
        Path(os.path.join(output_dir, "transcriptions_scoring", c_split)).mkdir(
            parents=True, exist_ok=True
        )
        if c_split.startswith("train"):
            ann_json = glob.glob(os.path.join(corpus_dir, "splits", c_split, "*.json"))
            list_file = os.path.join(corpus_dir, "splits", c_split + ".list")
        elif c_split == "dev":
            use_version = "_a"  # alternative version is _b see data section
            ann_json = glob.glob(
                os.path.join(corpus_dir, "splits", "dev" + use_version, "*.json")
            )
            list_file = os.path.join(corpus_dir, "splits", c_split + ".list")

        elif c_split == "eval":
            ann_json = glob.glob(os.path.join(corpus_dir, "splits", "test", "*.json"))
            list_file = os.path.join(corpus_dir, "splits", "test.list")

        sess2subintv = read_list_file(list_file)
        to_uem = []
        for j_file in ann_json:
            with open(j_file, "r") as f:
                annotation = json.load(f)
            sess_name = Path(j_file).stem
            # add session name
            # retrieve speakers from .list file

            subject, interviewer = sess2subintv[sess_name]

            [x.update({"session_id": sess_map[sess_name]}) for x in annotation]
            [x.update({"speaker": spk_map[x["speaker"]]}) for x in annotation]

            annotation, annotation_scoring = normalize_mixer6(
                annotation, scoring_txt_normalization
            )
            # create symlinks for audio,
            # note that we have to handle close talk here correctly

            create_audio_symlinks(
                c_split,
                sess_map[sess_name],
                sess2audio[sess_name],
                output_dir,
                spk_map[interviewer],
                spk_map[subject],
            )

            with open(
                os.path.join(
                    output_dir,
                    "transcriptions",
                    c_split,
                    sess_map[sess_name] + ".json",
                ),
                "w",
            ) as f:
                json.dump(annotation, f, indent=4)
            with open(
                os.path.join(
                    output_dir,
                    "transcriptions_scoring",
                    c_split,
                    sess_map[sess_name] + ".json",
                ),
                "w",
            ) as f:
                json.dump(annotation_scoring, f, indent=4)
            # dump uem too for dev only
            if c_split == "dev":
                uem_start = sorted(
                    annotation_scoring, key=lambda x: float(x["start_time"])
                )[0]["start_time"]
                uem_end = sorted(
                    annotation_scoring, key=lambda x: float(x["end_time"])
                )[-1]["end_time"]
                c_uem = "{} 1 {} {}\n".format(
                    sess_map[sess_name],
                    "{:.3f}".format(float(uem_start)),
                    "{:.3f}".format(float(uem_end)),
                )
                to_uem.append(c_uem)
            elif c_split == "eval":
                uem_start = 0
                uem_end = max([sf.SoundFile(x).frames for x in sess2audio[sess_name]])
                c_uem = "{} 1 {} {}\n".format(
                    sess_map[sess_name],
                    "{:.3f}".format(float(uem_start)),
                    "{:.3f}".format(float(uem_end / 16000)),
                )
                to_uem.append(c_uem)

        if len(to_uem) > 0:
            assert c_split in ["dev", "eval"]  # uem only for development set
            Path(os.path.join(output_dir, "uem", c_split)).mkdir(parents=True)
            to_uem = sorted(to_uem)
            with open(os.path.join(output_dir, "uem", c_split, "all.uem"), "w") as f:
                f.writelines(to_uem)
