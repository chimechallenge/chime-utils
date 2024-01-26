import glob
import json
import logging
import os
from pathlib import Path

from chime_utils.dgen.azure_storage import download_meeting_subset
from chime_utils.dgen.utils import get_mappings
from chime_utils.text_norm import get_txt_norm

logging.basicConfig(
    format=(
        "%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d]" " %(message)s"
    ),
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

NOTSOFAR1_FS = 16000


def download_notsofar1(download_dir, subset_name):
    if subset_name == "dev":
        subset_name = "dev_set"
        version = "240121_dev"
    dev_meetings_dir = download_meeting_subset(
        subset_name=subset_name, version=version, destination_dir=str(download_dir)
    )
    if dev_meetings_dir is None:
        logger.error(f"Failed to download {subset_name} for NOTSOFAR1 dataset")

    return os.path.join(download_dir, subset_name, version, "MTG")


def convert2chime(
    c_split, audio_dir, session_name, spk_map, txt_normalization, output_root
):
    output_audio_f = os.path.join(output_root, "audio", c_split)

    os.makedirs(output_audio_f, exist_ok=True)
    output_txt_f = os.path.join(output_root, "transcriptions", c_split)
    os.makedirs(output_txt_f, exist_ok=True)

    output_txt_f_norm = os.path.join(output_root, "transcriptions_scoring", c_split)
    os.makedirs(output_txt_f_norm, exist_ok=True)

    far_field_audio = glob.glob(os.path.join(audio_dir, "*.wav"))
    for elem in far_field_audio:
        # create symbolic link
        filename = Path(elem).stem
        tgt_name = os.path.join(
            output_audio_f,
            "{}_U01_CH{}.wav".format(session_name, int(filename.strip("ch")) + 1),
        )
        os.symlink(elem, tgt_name)

    if c_split.startswith("eval"):
        return  # no close talk and transcriptions
    close_talk_audio = glob.glob(
        os.path.join(Path(audio_dir).parent, "close_talk", "*.wav")
    )
    for elem in close_talk_audio:
        filename = Path(elem).stem
        tgt_name = os.path.join(
            output_audio_f,
            "{}_P{:02d}.wav".format(session_name, int(filename.split("_")[-1])),
        )
        os.symlink(elem, tgt_name)

    # load now transcription JSON and make some modifications
    with open(os.path.join(Path(audio_dir).parent, "gt_transcription.json"), "r") as f:
        transcriptions = json.load(f)

    output = []
    output_normalized = []
    for entry in transcriptions:
        # add sess id
        entry["session_id"] = session_name
        # convert floats to str
        entry["start_time"] = str(entry["start_time"])
        entry["end_time"] = str(entry["end_time"])
        entry["speaker"] = spk_map[entry["speaker_id"]]
        del entry["speaker_id"]
        entry["word_timing"] = [
            [x[0], str(x[1]), str(x[2])] for x in entry["word_timing"]
        ]
        output.append(entry)

        entry["words"] = txt_normalization(entry["words"])
        if len(entry["words"]) == 0:
            continue

        del entry["word_timing"]
        del entry["ct_wav_file_name"]
        output_normalized.append(entry)
        #

    output = sorted(output, key=lambda x: float(x["start_time"]))

    with open(os.path.join(output_txt_f, f"{session_name}.json"), "w") as f:
        json.dump(output, f, indent=4)


def prep_notsofar1(
    output_dir, corpus_dir, dset_part="dev", download=False, challenge="chime8"
):
    corpus_dir = Path(corpus_dir).resolve()  # allow for relative path
    mapping = get_mappings(challenge)
    spk_map = mapping["spk_map"]["dipco"]
    sess_map = mapping["sessions_map"]["dipco"]
    text_normalization = get_txt_norm(challenge)

    if download:
        corpus_dir = download_notsofar1(corpus_dir, subset_name=dset_part)
    else:
        if dset_part == "dev":
            subset_name = "dev_set"
            version = "240121_dev"
            corpus_dir = os.path.join(corpus_dir, subset_name, version, "MTG")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    # we fetch the
    device_jsons = glob.glob(
        os.path.join(corpus_dir, "**/devices.json"), recursive=True
    )
    device_jsons = sorted(device_jsons, key=lambda x: Path(x).parent.stem)

    for device_j in device_jsons:
        orig_sess_name = Path(device_j).parent.stem

        with open(device_j, "r") as f:
            devices_info = json.load(f)

        mc_devices = [
            x
            for x in devices_info
            if x["is_close_talk"] is False and x["is_mc"] is True
        ]

        for mc_device in mc_devices:
            device_folder = os.path.join(
                Path(device_j).parent, f"mc_{mc_device['device_name']}"
            )
            if not os.path.exists(device_folder):
                logging.warning(
                    f"Can't locate any directory for "
                    f"{mc_device['device_name']} in {orig_sess_name} folder."
                )
                continue
            device_name = mc_device["device_name"]
            sess_name = sess_map[f"{orig_sess_name}_{device_name}_mc"]
            convert2chime(
                dset_part,
                device_folder,
                sess_name,
                spk_map,
                text_normalization,
                output_dir,
            )
