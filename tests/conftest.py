def pytest_assertion_pass(item, lineno : int, orig : str, expl : str):
    print(f"{item} passed. Explanation: {expl}")