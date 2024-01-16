import pytest

from chime_utils.text_norm.whisper_like import EnglishTextNormalizer


@pytest.mark.parametrize("std", [EnglishTextNormalizer()])
def test_text_normalizer(std):
    assert std("Let's") == "let us"
    assert std("he's like") == "he is like"
    assert std("she's been like") == "she has been like"
    assert (
        std("Mr. Park visited Assoc. Prof. Kim Jr.")
        == "mister park visited associate professor kim junior"
    )
    assert (
        std("oh [unintelligible] one of the most common errors is")
        == "oh one of the most common errors is"
    )
    assert (
        std("o one of the most common errors is")
        == "o one of the most common errors is"
    )
    assert std("goin") == "going"
    assert std("uhhh") == "uh"
    assert std("aha") == "aha"
    assert std("one hundred") == "one hundred"
    assert std("one") == "one"
    assert std("ohhh wow") == "oh wow"
    assert std("ah") == "ah"
    assert std("wi-fi") == "wifi"
    assert std("wi fi") == "wifi"
    assert std("ummm hmmm") == "hmm hmm"
    assert std("uhh uhm") == "uh hmm"
    assert std("ahh okay") == "ah okay"
    assert (
        std("like five hundred dollars [unintelligible] per head [unintelligible]")
        == "like five hundred dollars per head"
    )
    assert (
        std("hmmm this is not as bad [unintelligible] ummm probably thirty minutes")
        == "hmm this is not as bad hmm probably thirty minutes"
    )
