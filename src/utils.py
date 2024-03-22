import os
import asyncio
import typing as tp


class Logger:
    def __init__(self, marker: str = "predict-timings"):
        pass


def get_env_var_or_default(var_name, default_value):
    env_value = os.environ.get(var_name, "")
    if len(env_value) > 0:
        return env_value
    return default_value


def get_loop() -> asyncio.AbstractEventLoop:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.new_event_loop()


def check_files_exists(remote_files: list[str], local_path: str) -> list[str]:
    local_files = os.listdir(local_path)

    missing_files = list(set(remote_files) - set(local_files))

    return missing_files


async def download_files_with_pget(
    remote_path: str, path: str, files: list[str]
) -> None:
    download_jobs = "\n".join(f"{remote_path}/{f} {path}/{f}" for f in files)
    args = ["pget", "multifile", "-", "-f", "--max-conn-per-host", "100"]
    process = await asyncio.create_subprocess_exec(*args, stdin=-1, close_fds=True)
    await process.communicate(download_jobs.encode())


def maybe_download_with_pget(
    path: str,
    remote_path: tp.Optional[str] = None,
    remote_filenames: tp.Optional[list[str]] = None,
    logger: tp.Optional[Logger] = None,
):
    if remote_path:
        remote_path = remote_path.rstrip("/")
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            missing_files = remote_filenames or []
        else:
            missing_files = check_files_exists(remote_filenames or [], path)
        get_loop().run_until_complete(
            download_files_with_pget(remote_path, path, missing_files)
        )

    return path
