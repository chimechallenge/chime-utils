"""
legacy CHiME-7 DASR and CHiME-6 text normalization.
"""

import jiwer
from jiwer.transforms import RemoveKaldiNonWords
from lhotse.recipes.chime6 import normalize_text_chime6

jiwer_chime6_scoring = jiwer.Compose(
    [
        RemoveKaldiNonWords(),
        jiwer.SubstituteRegexes(
            {r"\"": " ", "^[ \t]+|[ \t]+$": "", r"\u2019": "'"}
        ),  # noqa E501
        jiwer.RemoveEmptyStrings(),
        jiwer.RemoveMultipleSpaces(),
    ]
)
jiwer_chime7_scoring = jiwer.Compose(
    [
        jiwer.SubstituteRegexes(
            {
                "(?:^|(?<= ))(hm|hmm|mhm|mmh|mmm)(?:(?= )|$)": "hmmm",
                "(?:^|(?<= ))(uhm|um|umm|umh|ummh)(?:(?= )|$)": "ummm",
                "(?:^|(?<= ))(uh|uhh)(?:(?= )|$)": "uhhh",
            }
        ),
        jiwer.RemoveEmptyStrings(),
        jiwer.RemoveMultipleSpaces(),
    ]
)


def chime6_norm_scoring(txt):
    return jiwer_chime6_scoring(normalize_text_chime6(txt, normalize="kaldi"))


def chime7_norm_scoring(txt):
    """
    here we also normalize non-words sounds such as hmmm which are quite a lot ! # noqa 501
    you are free to use whatever normalization you prefer for training but this
    normalization below will be used when we score your submissions.
    """
    return jiwer_chime7_scoring(
        jiwer_chime6_scoring(
            normalize_text_chime6(txt, normalize="kaldi")
        )  # noqa: E731
    )  # noqa: E731
