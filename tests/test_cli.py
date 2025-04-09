import os
import subprocess


def test_smoke_test():
    cmd = ["python", "-m", "swebench.harness.run_evaluation", "--help"]
    result = subprocess.run(cmd, capture_output=True)
    print(result.stdout)
    print(result.stderr)
    assert result.returncode == 0


def test_one_instance():
    cmd = [
        "python",
        "-m",
        "swebench.harness.run_evaluation",
        "--predictions_path",
        "gold",
        "--max_workers",
        "1",
        "--instance_ids",
        "sympy__sympy-20590",
        "--run_id",
        "validate-gold",
    ]
    result = subprocess.run(cmd, capture_output=True)
    print(result.stdout)
    print(result.stderr)
    assert result.returncode == 0

def test_run_api():
    os.environ.setdefault("OPENAI_API_KEY", "")
    cmd = [
        "python",
        "-m",
        "swebench.inference.run_api",
        "--dataset_name_or_path",
        "princeton-nlp/SWE-bench_oracle",
        "--model_name_or_path",
        "claude-2",
        "--output_dir",
        "./outputs",
    ]
    result = subprocess.run(cmd, capture_output=True)
    print(result.stdout)
    print(result.stderr)
    assert result.returncode == 0

def test_run_live():
    os.environ.setdefault("GITHUB_TOKEN", "xxx")
    cmd = [
        "python",
        "-m",
        "swebench.inference.run_live",
        "--model_name",
        "gpt-3.5-turbo-1106",
        "--issue_url",
        "https://github.com/huggingface/transformers/issues/26706",
    ]
    result = subprocess.run(cmd, capture_output=True)
    print(result.stdout)
    print(result.stderr)
    assert result.returncode == 0


if __name__ == '__main__':
    # test_one_instance()
    # test_run_api()
    test_run_live()



