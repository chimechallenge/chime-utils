import pytest

from chime_utils.text_norm.whisper_like import EnglishTextNormalizer


@pytest.mark.parametrize("std", [EnglishTextNormalizer()])
def test_text_normalizer(std):
    assert std("shan't") == "shall not"
    assert std("han't") == "has not"
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
    assert std("uhhh") == ""
    assert std("et voila") == "et voila"
    assert std("huh") == ""
    assert std("aha") == "aha"
    assert std("one hundred") == "one hundred"
    assert std("one") == "one"
    assert std("ohhh wow") == "oh wow"
    assert std("ah") == ""
    assert std("wi-fi") == "wifi"
    assert std("wi fi") == "wifi"
    assert std("ummm hmmm") == ""
    assert std("uhh oh") == "oh"
    assert std("ahh okay") == "ok"
    assert std("okay") == "ok"
    assert std("20$") == "twenty dollars"
    assert (
        std(
            "uh well hopefully the process is still going on aaron but uh in the about uh forty minutes that way and about thirty years ago i moved into the city and uh"
        )
        == "well hopefully the process is still going on aaron but in the about forty minutes that way and about thirty years ago i moved into the city and"
    )
    assert (
        std("like five hundred dollars [unintelligible] per head" " [unintelligible]")
        == "like five hundred dollars per head"
    )
    assert (
        std("hmmm this is not as bad [unintelligible] ummm probably thirty" " minutes")
        == "this is not as bad probably thirty minutes"
    )
