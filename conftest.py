def pytest_assertion_pass(item: function, lineno : int, orig : str, expl : str):
    print(f"{item} passed. Explanation: {expl}")