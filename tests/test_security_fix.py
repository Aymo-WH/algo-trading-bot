import subprocess
import os
import sys

def run_inference(config_val):
    result = subprocess.run(
        [sys.executable, "src/live_inference.py", "--config", config_val],
        capture_output=True,
        text=True
    )
    return result.stdout + result.stderr

def test_security():
    print("Testing Security Fix for Path Traversal...")

    # 1. Test legitimate path in config/
    print("\n[TEST 1] Legitimate config (config.json)")
    output = run_inference("config.json")
    if "[SYSTEM] Booting Live Inference Engine..." in output and "[ERROR] Security:" not in output:
        print("PASS: Legitimate config handled correctly.")
    else:
        print(f"FAIL: Legitimate config failed. Output:\n{output}")

    # 2. Test legitimate path with full path
    print("\n[TEST 2] Legitimate full path (config/config.json)")
    output = run_inference("config/config.json")
    if "[SYSTEM] Booting Live Inference Engine..." in output and "[ERROR] Security:" not in output:
        print("PASS: Legitimate full path handled correctly.")
    else:
        print(f"FAIL: Legitimate full path failed. Output:\n{output}")

    # 3. Test Path Traversal Attempt
    print("\n[TEST 3] Path Traversal Attempt (../README.md)")
    output = run_inference("../README.md")
    if "[ERROR] Security: Configuration path" in output:
        print("PASS: Path traversal attempt correctly blocked.")
    else:
        print(f"FAIL: Path traversal attempt NOT blocked. Output:\n{output}")

    # 4. Test Absolute Path Traversal
    print("\n[TEST 4] Absolute Path Traversal (/etc/passwd)")
    output = run_inference("/etc/passwd")
    if "[ERROR] Security: Configuration path" in output:
        print("PASS: Absolute path traversal attempt correctly blocked.")
    else:
        print(f"FAIL: Absolute path traversal attempt NOT blocked. Output:\n{output}")

    # 5. Test another traversal
    print("\n[TEST 5] Complex Traversal (config/../README.md)")
    output = run_inference("config/../README.md")
    if "[ERROR] Security: Configuration path" in output:
        print("PASS: Complex traversal attempt correctly blocked.")
    else:
        print(f"FAIL: Complex traversal attempt NOT blocked. Output:\n{output}")

if __name__ == "__main__":
    test_security()
