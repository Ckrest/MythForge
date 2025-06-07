import os
import subprocess

REQ_DIR = os.path.join(os.path.dirname(__file__), 'requirements')
REQ_FILE = os.path.join(REQ_DIR, 'requirements.txt')


def choose_python_command():
    """Return a python command ('python' or 'py') that exists."""
    for cmd in ("python", "py"):
        try:
            result = subprocess.run([cmd, "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode == 0:
                return cmd
        except FileNotFoundError:
            continue
    return None


def download_package(cmd, package):
    subprocess.check_call([cmd, "-m", "pip", "download", package, "-d", REQ_DIR])


def main():
    os.makedirs(REQ_DIR, exist_ok=True)
    python_cmd = choose_python_command()
    if not python_cmd:
        raise RuntimeError("Neither 'python' nor 'py' command was found")

    with open(REQ_FILE) as f:
        for line in f:
            pkg = line.strip()
            if not pkg or pkg.startswith('#'):
                continue
            download_package(python_cmd, pkg)


if __name__ == "__main__":
    main()
