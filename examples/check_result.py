import os
import numpy as np


golden = set()
tests = set()


for file in os.listdir("."):
    if file.endswith("_golden.bin"):
        key = file.replace("_golden.bin", "")
        golden.add(key)
    if file.endswith("_test.bin"):
        key = file.replace("_test.bin", "")
        tests.add(key)

for key in golden:
    if key in tests:
        golden_file = f"{key}_golden.bin"
        test_file = f"{key}_test.bin"
        golden_data = np.fromfile(golden_file, dtype=np.float32)
        test_data = np.fromfile(test_file, dtype=np.float32)
        if np.allclose(golden_data, test_data, atol=1e-5):
            print(f"{key}: PASS")
        else:
            print(f"{key}: FAIL")
        tests.remove(key)
    else:
        print(f"{key}: MISSING TEST FILE")

for key in tests:
    print(f"{key}: MISSING GOLDEN FILE")