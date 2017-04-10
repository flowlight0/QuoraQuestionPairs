from stopwords import build_anchors, process, fix


def test_just_works():
    rules = []
    anchors = build_anchors(rules)
    stopwords = {"a", "the"}
    result = process(stopwords, anchors, fix("what is the best question on quora?"))
    expected = {tuple("what is best question on quora".split())}
    assert expected == result

def test_does_something():
    rules = [(tuple("what is".split()), tuple("which is".split()))]
    anchors = build_anchors(rules)
    stopwords = {"a", "the"}
    result = process(stopwords, anchors, fix("what is the best question on quora?"))
    expected = {
        tuple("what is best question on quora".split()),
        tuple("which is best question on quora".split())
    }
    assert expected == result

def test_completes():
    rules = [(tuple("best".split()), tuple("best best".split()))]
    anchors = build_anchors(rules)
    stopwords = {"a", "the"}
    result = process(stopwords, anchors, fix("what is the best question on quora?"))
    expected = {
        tuple("what is best question on quora".split()),
        tuple("what is best best question on quora".split()),
        tuple("what is best best best question on quora".split()),
        tuple("what is best best best best question on quora".split()),
        tuple("what is best best best best best question on quora".split()),
        tuple("what is best best best best best best question on quora".split())
    }
    assert expected == result
