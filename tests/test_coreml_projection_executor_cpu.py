"""Exercise ownership and transport contracts without importing Core ML/MLX."""

import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from mlx2.runtime import ane_projection as ap

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def probe():
    spec = importlib.util.spec_from_file_location("coreml_probe", ROOT / "scripts/probe_coreml_projection.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def compiled(tmp_path):
    path = tmp_path / "original.mlmodelc"
    (path / "weights").mkdir(parents=True)
    (path / "model.mil").write_bytes(b"reviewed synthetic model")
    (path / "weights/weight.bin").write_bytes(b"reviewed synthetic weights")
    return path, ap.compiled_model_digest(path)


@pytest.fixture
def runtime(monkeypatch):
    state = NS(loads=[], predictions=[], plans=[], output=None, enter=None, release=None)
    weights = np.arange(8, dtype=np.float16).reshape(2, 4) / 8
    policy = object()

    class Model:
        def __init__(self, path, *, compute_units):
            assert compute_units is policy
            state.loads.append(Path(path))

        def predict(self, values):
            state.predictions.append(values)
            if state.enter is not None:
                state.enter.set()
                assert state.release.wait(5)
            if state.output is not None:
                return state.output
            return {"projection": (values["x"] @ weights.T).astype(np.float16)}

    cpu = type("MLCPUComputeDevice", (), {})()
    ne = type("MLNeuralEngineComputeDevice", (), {})()
    nested = NS(operator_name="linear", blocks=[], device=ne)
    outer = NS(operator_name="cast", blocks=[NS(operations=[nested])], device=cpu)
    state.operations = [outer]

    class Plan:
        @classmethod
        def load_from_path(cls, path, *, compute_units):
            assert compute_units is policy
            state.plans.append(Path(path))
            return NS(
                model_structure=NS(program=NS(functions={"main": NS(block=NS(operations=state.operations))})),
                get_compute_device_usage_for_mlprogram_operation=lambda op: NS(preferred_compute_device=op.device),
            )

    runtime = NS(models=NS(CompiledMLModel=Model, MLComputePlan=Plan),
                 ComputeUnit=NS(CPU_AND_NE=policy), __version__="fake-cpu-contract-only")
    monkeypatch.setattr(ap, "_coreml_runtime", lambda: runtime)
    return state


def make_executor(compiled):
    path, digest = compiled
    spec = ap.FixedProjection("test.adapter", hashlib.sha256(b"fixture").hexdigest(), 4, 2, 1)
    return ap.CoreMLProjectionExecutor(spec, path, declared_sha256=digest)


def test_prediction_retains_reviewed_bytes_and_plan_never_proves_ane(compiled, runtime):
    source, digest = compiled
    with make_executor(compiled) as executor:
        retained = runtime.loads[0]
        assert retained != source and ap.compiled_model_digest(retained) == digest
        (source / "model.mil").write_bytes(b"changed after admission")
        assert ap.compiled_model_digest(retained) == digest
        x = np.array([[1, 2, 3, 4]], dtype=np.float16)
        output, receipt = executor.predict(x)
        assert output.tolist() == [[2.5, 7.5]]
        assert not output.flags.writeable
        assert runtime.predictions[0]["x"] is not x
        assert receipt["prediction_completed"] is True
        assert receipt["selected"] is True
        assert receipt["qualified"] is receipt["observed_used"] is False
        assert receipt["execution_device_observed"] is False
        assert receipt["compiled_model_sha256"] == digest
        plan = executor.compute_plan()
        assert plan["selected"] is plan["prediction_completed"] is True
        assert plan["prediction_count"] == 1
        assert [op["anticipated_device"] for op in plan["compute_plan"]] == ["cpu", "neural_engine"]
        assert plan["observed_used"] is plan["execution_device_observed"] is False
        assert runtime.plans == [retained]
    assert not retained.exists()
    executor.close()
    with pytest.raises(RuntimeError, match="closed"):
        executor.predict(x)


@pytest.mark.parametrize("value", [
    [[1, 2, 3, 4]], np.ones((1, 4), dtype=np.float32), np.ones((2, 4), dtype=np.float16),
    np.array([[1, 2, 3, np.inf]], dtype=np.float16),
])
def test_invalid_input_fails_before_native_predict(compiled, runtime, value):
    with make_executor(compiled) as executor, pytest.raises((TypeError, ValueError), match="input"):
        executor.predict(value)
    assert not runtime.predictions


@pytest.mark.parametrize("output", [
    {}, {"projection": np.ones((1, 2), dtype=np.int32)},
    {"projection": np.ones((2, 2), dtype=np.float16)},
    {"projection": np.array([[1, np.nan]], dtype=np.float16)},
])
def test_invalid_native_output_cannot_yield_success_receipt(compiled, runtime, output):
    runtime.output = output
    with make_executor(compiled) as executor:
        with pytest.raises(ValueError):
            executor.predict(np.ones((1, 4), dtype=np.float16))
        # Native completion is distinct from validated numerical output.
        assert executor.compute_plan()["prediction_completed"] is True


def test_coreml_bridge_float32_output_is_bounded_back_to_fp16(compiled, runtime):
    runtime.output = {"projection": np.array([[1.25, -2.5]], dtype=np.float32)}
    with make_executor(compiled) as executor:
        output, receipt = executor.predict(np.ones((1, 4), dtype=np.float16))
    assert output.dtype == np.float16
    assert output.tolist() == [[1.25, -2.5]]
    assert receipt["native_output_dtype"] == "float32"
    assert receipt["returned_output_dtype"] == "float16"


def test_copy_changed_after_review_is_rejected_before_native_load(compiled, runtime, monkeypatch):
    copy = ap._copy_compiled_tree

    def changed(source, destination):
        copy(source, destination)
        target = destination / "model.mil"
        target.chmod(0o600)
        target.write_bytes(b"different bytes")

    monkeypatch.setattr(ap, "_copy_compiled_tree", changed)
    with pytest.raises(ValueError, match="reviewed identity"):
        make_executor(compiled)
    assert not runtime.loads


def test_root_symlink_rejected_before_native_load(compiled, runtime, tmp_path):
    original, digest = compiled
    link = tmp_path / "link.mlmodelc"
    link.symlink_to(original, target_is_directory=True)
    with pytest.raises(ValueError, match="non-symlink"):
        make_executor((link, digest))
    assert not runtime.loads


def test_model_load_failure_removes_private_snapshot(compiled, runtime, monkeypatch):
    paths = []

    def failed(path, **_kwargs):
        paths.append(Path(path))
        raise RuntimeError("native load failed")

    fake = ap._coreml_runtime()
    monkeypatch.setattr(fake.models, "CompiledMLModel", failed)
    with pytest.raises(RuntimeError, match="native load failed"):
        make_executor(compiled)
    assert len(paths) == 1 and not paths[0].exists()


def test_close_retains_native_model_and_files_until_prediction_drains(compiled, runtime):
    runtime.enter, runtime.release = threading.Event(), threading.Event()
    executor = make_executor(compiled)
    retained = runtime.loads[0]
    outputs, errors = [], []

    def predict():
        try:
            outputs.append(executor.predict(np.ones((1, 4), dtype=np.float16)))
        except Exception as exc:  # noqa: BLE001 - preserve worker failures for the parent test
            errors.append(exc)

    predictor = threading.Thread(target=predict)
    closer = threading.Thread(target=executor.close)
    predictor.start()
    try:
        assert runtime.enter.wait(3)
        closer.start()
        closer.join(.05)
        assert closer.is_alive() and retained.exists()
    finally:
        runtime.release.set()
        predictor.join(5)
        if closer.ident is not None:
            closer.join(5)
        executor.close()
    assert not predictor.is_alive() and not closer.is_alive()
    assert len(outputs) == 1 and not errors and not retained.exists()


def test_plan_unknown_is_not_claim_and_gpu_is_contract_failure(compiled, runtime):
    with make_executor(compiled) as executor:
        runtime.operations = [NS(operator_name="unknown", blocks=[], device=None)]
        plan = executor.compute_plan()
        assert plan["compute_plan"][0]["anticipated_device"] == "unknown"
        assert plan["prediction_completed"] is plan["observed_used"] is False
        runtime.operations[0].device = type("MLGPUComputeDevice", (), {})()
        with pytest.raises(ValueError, match="GPU"):
            executor.compute_plan()


def test_import_is_accelerator_runtime_free():
    command = "import sys; import mlx2.runtime.ane_projection; assert not any(n.split('.')[0] in {'mlx', 'torch', 'coremltools'} for n in sys.modules)"
    result = subprocess.run([sys.executable, "-c", command], capture_output=True, text=True, timeout=10, check=False)
    assert result.returncode == 0, result.stderr


def test_probe_fixture_has_complete_finite_reference(probe):
    weights, inputs, recipe, digest = probe.fixture()
    assert recipe["shape"] == [2, 8, 4] and len(digest) == 64
    assert len(inputs) == 3
    for values in inputs:
        reference = values.astype(np.float32) @ weights.astype(np.float32).T
        assert values.dtype == weights.dtype == np.float16
        assert reference.shape == (2, 4) and np.isfinite(reference).all()


@pytest.mark.parametrize("command,status", [
    (["-c", "print('positive control')"], "completed"),
    (["-c", "raise SystemExit(7)"], "worker_failed"),
    (["-c", "import time; time.sleep(20)"], "deadline"),
])
def test_supervisor_bounds_cpu_child_and_reaps(probe, tmp_path, command, status):
    before = signal.getsignal(signal.SIGTERM)
    receipt = probe.supervise([sys.executable, *command], timeout=.5, output=tmp_path)
    assert receipt["status"] == status
    assert receipt["worker_exit_confirmed"] is True
    assert receipt["elapsed_seconds"] < 5
    assert signal.getsignal(signal.SIGTERM) == before


def test_supervisor_term_is_cleanup_not_orphan(probe, tmp_path):
    script = tmp_path / "supervisor.py"
    output = tmp_path / "out"
    output.mkdir()
    pid_path = tmp_path / "worker.pid"
    script.write_text(
        "import importlib.util,json,sys\nfrom pathlib import Path\n"
        f"s=importlib.util.spec_from_file_location('probe',{str(ROOT / 'scripts/probe_coreml_projection.py')!r})\n"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m)\n"
        f"r=m.supervise([sys.executable,'-c',\"import os,time;from pathlib import Path;Path({str(pid_path)!r}).write_text(str(os.getpid()));time.sleep(20)\"],timeout=10,output=Path({str(output)!r}))\n"
        f"Path({str(tmp_path / 'result.json')!r}).write_text(json.dumps(r))\n"
    )
    process = subprocess.Popen([sys.executable, str(script)])
    try:
        deadline = time.monotonic() + 5
        while not pid_path.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert pid_path.exists()
        worker = int(pid_path.read_text())
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=5) == 0
        receipt = json.loads((tmp_path / "result.json").read_text())
        assert receipt["status"] == "interrupted"
        assert receipt["interrupted_signal"] == signal.SIGTERM
        assert receipt["worker_exit_confirmed"] is True
        with pytest.raises(ProcessLookupError):
            os.kill(worker, 0)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_existing_evidence_is_refused_before_ownership_or_runtime(probe, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "qualification_gpu_ownership", None)
    with pytest.raises(SystemExit) as exc:
        probe.main(["--output-dir", str(tmp_path), "--cpg-task", "unused",
                    "--cpg-owner-lock", "unused", "--cpg-generation", "1"])
    assert exc.value.code == 2
