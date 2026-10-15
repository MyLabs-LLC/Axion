from __future__ import annotations

import os
import subprocess
from pathlib import Path


def test_publisher_exports_from_the_checked_hosts_and_root(tmp_path: Path) -> None:
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    checkout = tmp_path / "alternate checkout"
    checkout.mkdir()
    for name, body in {
        "ssh": "#!/bin/sh\nexit 0\n",
        "uv": "#!/bin/sh\nprintf 'export:%s\\n' \"$@\"\n",
        "hf": "#!/bin/sh\nexit 0\n",
    }.items():
        executable = tools_dir / name
        executable.write_text(body, encoding="utf-8")
        executable.chmod(0o755)

    script = Path(__file__).resolve().parents[1] / "scripts" / "publish_hf_checkpoint.sh"
    hosts = ["stage-zero", "stage-one", "stage-two"]
    result = subprocess.run(
        [
            "bash",
            str(script),
            "--run-name",
            "test-run",
            "--step",
            "2000",
            "--repo",
            "test-owner/test-model",
            "--root",
            str(checkout),
            "--hosts",
            *hosts,
            "--uv-bin",
            str(tools_dir / "uv"),
            "--hf-bin",
            str(tools_dir / "hf"),
        ],
        env={**os.environ, "PATH": f"{tools_dir}{os.pathsep}{os.environ['PATH']}"},
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    exported_args = [
        line.removeprefix("export:")
        for line in result.stdout.splitlines()
        if line.startswith("export:")
    ]
    root_index = exported_args.index("--root")
    assert exported_args[root_index + 1] == str(checkout.resolve())
    hosts_index = exported_args.index("--hosts")
    assert exported_args[hosts_index + 1 : hosts_index + 4] == hosts
