import os
from typing import Dict, Optional, Union

from lhotse.audio import RecordingSet
from lhotse.kaldi import export_to_kaldi
from lhotse.supervision import SupervisionSet
from lhotse.utils import Pathlike

import chime_utils.dprep.lhotse as lhotse_prep


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
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Creates Kaldi-style manifests for CHiME-6.
    :param corpus_dir: Pathlike, the path of CHiME-6 main directory.
    :param output_dir: Pathlike, the path where to write the manifests.
    :param mic: str, the microphone type to use,
    choose from "ihm" (close-talk) or "mdm"
        (multi-microphone array) settings.
        For MDM, there are 6 array devices with 4 channels each,
        so the resulting recordings will have 24 channels (for most sessions).
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
    """
    manifests = lhotse_prep.prepare_chime6(
        corpus_dir, None, dset_part, mic, json_dir, discard_problematic, txt_norm
    )
    # now we need to convert manifests to kaldi format via lhotse
    for k in manifests.keys():
        c_out_dir = os.path.join(output_dir, k)
        os.makedirs(c_out_dir, exist_ok=True)
        export_to_kaldi(
            manifests[k]["recordings"],
            manifests[k]["supervisions"],
            c_out_dir,
            prefix_spk_id=False,  # already appended
            map_underscores_to="-",
        )


def prepare_dipco(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    dset_part: Optional[str] = "dev",
    mic: Optional[str] = "mdm",
    json_dir: Optional[
        Pathlike
    ] = None,  # alternative annotation e.g. from non-oracle diarization
    txt_norm: Optional[str] = "chime8",
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Creates Kaldi-style manifests for DiPCo.
    :param corpus_dir: Pathlike, the path of DiPCo main directory.
    :param output_dir: Pathlike, the path where to write the manifests.
    :param mic: str, the microphone type to use,
    choose from "ihm" (close-talk) or "mdm"
        (multi-microphone array) settings.
        For MDM, there are 5 array devices with 7
        channels each, so the resulting recordings will have 35 channels.
    :param json_dir: Pathlike, override the JSON annotation directory
        of the current dataset partition (e.g. dev)
        this allows for example to create a manifest from for example a JSON
        created with forced alignment available at
         https://github.com/chimechallenge/CHiME7_DASR_falign.
    :param txt_norm: str, which text normalization preprocessing
        one wishes to use; choose between 'chime7' and 'chime8' or None.
    :return dict: Dict whose key is the dataset part
        ("train", "dev" and "eval"), and the
        value is Dicts with the keys 'recordings' and 'supervisions'.
    """
    manifests = lhotse_prep.prepare_dipco(
        corpus_dir, None, dset_part, mic, json_dir, txt_norm
    )
    # now we need to convert manifests to kaldi format via lhotse
    for k in manifests.keys():
        c_out_dir = os.path.join(output_dir, k)
        os.makedirs(c_out_dir, exist_ok=True)
        export_to_kaldi(
            manifests[k]["recordings"],
            manifests[k]["supervisions"],
            c_out_dir,
            prefix_spk_id=False,  # already appended
            map_underscores_to="-",
        )


def prepare_mixer6(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    dset_part="dev",
    mic: Optional[str] = "mdm",
    json_dir: Optional[
        Pathlike
    ] = None,  # alternative annotation e.g. from non-oracle diarization
    txt_norm: Optional[str] = "chime8",
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Creates Kaldi-style manifests for Mixer6.
    :param corpus_dir: Pathlike, the path of Mixer 6 Speech main directory.
    :param output_dir: Pathlike, the path where to write the manifests.
    :param mic: str, the microphone type to use,
    choose from "ihm" (close-talk) or "mdm"
        (multi-microphone array) settings.
        For MDM, there are 11 channels.
    :param txt_norm: str, which text normalization preprocessing
        one wishes to use; choose between 'chime7' and 'chime8' or None.
    :return dict: Dict whose key is the dataset part
    ("train", "dev" and "eval"), and the
        value is Dicts with the keys 'recordings' and 'supervisions'.
    """
    manifests = lhotse_prep.prepare_mixer6(
        corpus_dir, None, dset_part, mic, json_dir, txt_norm
    )
    # now we need to convert manifests to kaldi format via lhotse
    for k in manifests.keys():
        c_out_dir = os.path.join(output_dir, k)
        os.makedirs(c_out_dir, exist_ok=True)
        export_to_kaldi(
            manifests[k]["recordings"],
            manifests[k]["supervisions"],
            c_out_dir,
            prefix_spk_id=False,  # already appended
            map_underscores_to="-",
        )


def prepare_notsofar1(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    dset_part="dev",
    mic: Optional[str] = "mdm",
    json_dir: Optional[
        Pathlike
    ] = None,  # alternative annotation e.g. from non-oracle diarization
    txt_norm: Optional[str] = "chime8",
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Creates Kaldi-style manifests for NOTSOFAR1.
    :param corpus_dir: Pathlike, the path of NOTSOFAR1 main directory.
    :param output_dir: Pathlike, the path where to write the manifests.
    :param mic: str, the microphone type to use,
    choose from "ihm" (close-talk) or "mdm"
        (multi-microphone array) settings.
        For MDM, there are 11 channels.
    :param txt_norm: str, which text normalization preprocessing
        one wishes to use; choose between 'chime7' and 'chime8' or None.
    :return dict: Dict whose key is the dataset part
    ("train", "dev" and "eval"), and the
        value is Dicts with the keys 'recordings' and 'supervisions'.
    """
    manifests = lhotse_prep.prepare_notsofar1(
        corpus_dir, None, dset_part, mic, json_dir, txt_norm
    )
    # now we need to convert manifests to kaldi format via lhotse
    for k in manifests.keys():
        c_out_dir = os.path.join(output_dir, k)
        os.makedirs(c_out_dir, exist_ok=True)
        export_to_kaldi(
            manifests[k]["recordings"],
            manifests[k]["supervisions"],
            c_out_dir,
            prefix_spk_id=False,  # already appended
            map_underscores_to="-",
        )
