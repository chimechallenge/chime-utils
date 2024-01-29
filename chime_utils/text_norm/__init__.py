from chime_utils.text_norm.c7dasr import chime6_norm_scoring, chime7_norm_scoring
from chime_utils.text_norm.whisper_like import EnglishTextNormalizer


def get_txt_norm(txt_norm):
    assert txt_norm in ["chime6", "chime7", "chime8", None, "none", "None", ""]
    if txt_norm in [None, "none", "None", ""]:
        return None
    elif txt_norm == "chime8":
        return EnglishTextNormalizer()
    elif txt_norm == "chime7":
        return chime7_norm_scoring
    elif txt_norm == "chime6":
        return chime6_norm_scoring
    else:
        raise NotImplementedError
