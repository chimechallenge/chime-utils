import glob
import json
import logging
import os
from pathlib import Path
from typing import Optional

from lhotse.utils import Pathlike

from chime_utils.text_norm import get_txt_norm

logging.basicConfig(
    format=(
        "%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d]" " %(message)s"
    ),
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


CHIME_6_FS = 16000
DIPCO_FS = 16000
MIXER6_FS = 16000


def prepare_chime6(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    dset_part: str = "dev",
    mic: str = "mdm",
    json_dir: Optional[
        Pathlike
    ] = None,  # alternative annotation e.g. from non-oracle diarization
    discard_problematic: Optional[bool] = True,
    txt_norm: Optional[str] = "chime8",
):
    """
    Returns the Speechbrain JSON
    manifests which consist of the Recordings and Supervisions
    :param corpus_dir: Pathlike, the path of CHiME-6 main directory.
    :param output_dir: Pathlike, the path where to write the manifests.
    :param mic: str, the microphone type to use,
    choose from "ihm" (close-talk) or "mdm" (multi-mic array) settings.
        For MDM, there are 6 array devices with 4 channels each,
        so the resulting recordings will have
        24 channels (for most sessions).
    :param json_dir: Pathlike, override the JSON annotation directory
        of the current dataset partition (e.g. dev)
        this allows for example to create a manifest from for example a JSON
        created with forced alignment available at
         https://github.com/chimechallenge/CHiME7_DASR_falign.
    :param discard_problematic: bool, whether discard problematic arrays
        in different sessions (arrays that have missing samples etc
        see https://chimechallenge.github.io/chime6/track1_data.html)
    :param txt_norm: str, which text normalization preprocessing
        one wishes to use; choose between 'chime7' and 'chime8' or None.
    :return dict: Dict, see https://arxiv.org/pdf/2106.04624.pdf section
         4.2. Speechbrain JSON annotation format for long-form audio.
    """
    txt_normalizer = get_txt_norm(txt_norm)
    assert mic in ["ihm", "mdm"], "mic must be either 'ihm' or 'mdm'."

    transcriptions_dir = (
        os.path.join(corpus_dir, "transcriptions_scoring")
        if json_dir is None
        else json_dir
    )

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    all_sessions = [
        Path(x).stem
        for x in glob.glob(os.path.join(transcriptions_dir, dset_part, "*.json"))
    ]

    manifest = {}
    # utt_id will be like
    for session in all_sessions:
        with open(os.path.join(transcriptions_dir, dset_part, f"{session}.json")) as f:
            transcript = json.load(f)
            for idx, segment in enumerate(transcript):
                spk_id = segment["speaker"]
                start = float(segment["start_time"])
                end = float(segment["end_time"])

                c_words = txt_normalizer(segment["words"])
                if len(c_words) == 0:
                    continue

                if start >= end:
                    raise RuntimeError(
                        "Current segment has negative duration ! "
                        "Something is wrong, exiting."
                        f"Current segment info: start: {start} end: "
                        f"{end} session: {session} speaker: {spk_id}"
                    )

                if mic == "ihm":
                    c_audio = os.path.join(
                        corpus_dir,
                        "audio",
                        dset_part,
                        f"{session}_{spk_id}.wav",
                    )
                    ex_id = (
                        f"{session}-{spk_id}-"
                        f"{round(start, 3)*100}-{round(end, 3)*100}-ihm"
                    )
                    manifest[ex_id] = {
                        "wav": {
                            "files": c_audio,
                            "start": int(start * CHIME_6_FS),
                            "stop": int(end * CHIME_6_FS),
                        },
                        "length": end - start,
                        "speaker": spk_id,
                        "words": c_words,
                    }
                elif mic == "mdm":
                    # get all far field microphones.
                    # we need to iterate on those
                    c_audios = glob.glob(
                        os.path.join(
                            corpus_dir,
                            "audio",
                            dset_part,
                            f"{session}_U0*.CH*.wav",
                        )
                    )

                    # discard problematic here
                    if discard_problematic and session == "S12":
                        c_audios = [
                            x
                            for x in c_audios
                            if not Path(x).stem.startswith("S12_U05")
                        ]
                    elif discard_problematic and session == "S24":
                        c_audios = [
                            x
                            for x in c_audios
                            if not Path(x).stem.startswith("S24_U06")
                        ]
                    elif discard_problematic and session == "S18":
                        c_audios = [
                            x
                            for x in c_audios
                            if not Path(x).stem.startswith("S18_U06")
                        ]

                    ex_id = (
                        f"{session}-{spk_id}-"
                        f"{round(start, 3) * 100}"
                        f"-{round(end, 3) * 100}-mdm"
                    )
                    manifest[ex_id] = {
                        "wav": {
                            "files": c_audios,
                            "start": int(start * CHIME_6_FS),
                            "stop": int(end * CHIME_6_FS),
                        },
                        "length": end - start,
                        "speaker": spk_id,
                        "words": c_words,
                    }

        with open(os.path.join(output_dir, f"chime6-{dset_part}-{mic}.json"), "w") as f:
            json.dump(manifest, f, indent=4)

        return manifest
