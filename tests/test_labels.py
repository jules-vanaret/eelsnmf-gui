from eelsnmf_gui._labels import split_common_label_parts


def test_single_label():
    """A single label maps to itself as common part and individual."""
    assert split_common_label_parts(["abc"]) == ("abc", ["abc"])


def test_empty_list():
    """An empty list yields an empty common part and no individuals."""
    assert split_common_label_parts([]) == ("", [])


def test_common_prefix_tokens_are_removed():
    """Tokens shared by all labels become the common part."""
    common, individuals = split_common_label_parts(["sample_A", "sample_B"])
    assert common == "sample"
    assert individuals == ["A", "B"]


def test_no_common_tokens():
    """Labels sharing no token give an empty common part."""
    common, individuals = split_common_label_parts(["x_1", "y_2"])
    assert common == ""
    assert individuals == ["x_1", "y_2"]


def test_partial_common_tokens():
    """Only leading tokens shared by every label count as common."""
    common, individuals = split_common_label_parts(
        ["a_b_c1", "a_b_c2", "a_b_c3"]
    )
    assert common == "a_b"
    assert individuals == ["c1", "c2", "c3"]


def test_uneven_token_counts():
    """Labels with different token counts still split without error."""
    common, individuals = split_common_label_parts(["a_b", "a_b_c"])
    assert individuals[0] == "" or individuals == ["", "c"]
