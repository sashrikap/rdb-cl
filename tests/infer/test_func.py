def test_func1():
    def func(a, b, c, d):
        return a + b + c + d

    args1 = {"a": 1, "b": 2}
    args2 = {"c": 1, "d": 2}
    out = func(**args1, **args2)
    print(out)
