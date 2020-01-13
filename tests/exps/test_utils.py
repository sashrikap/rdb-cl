from rdb.exps.utils import create_params


def test_params():
    template = {"a": 1, "b": 2}
    out_params = create_params(template, {"a": [1, 2], "b": [2, 3]})
    out_true = [{"a": 1, "b": 2}, {"a": 1, "b": 3}, {"a": 2, "b": 2}, {"a": 2, "b": 3}]
    for d in out_true:
        assert sum([d_ == d for d_ in out_params])

    template = {"a": 1, "b": 2}
    out_params = create_params(template, {"a": [], "b": []})
    assert len(out_params) == 1
    assert out_params[0] == template

    template = {"a": 1, "b": 2}
    out_params = create_params(template, {})
    assert len(out_params) == 1
    assert out_params[0] == template
