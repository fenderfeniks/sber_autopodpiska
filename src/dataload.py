import subprocess

def dvc_add(filepath: str):
    """Добавляет файл под контроль DVC и коммитит .dvc файл в git."""
    subprocess.run(["dvc", "add", filepath], check=True)
    subprocess.run(["git", "add", f"{filepath}.dvc", ".gitignore"], check=True)
    subprocess.run(["git", "commit", "-m", f"dvc: add {filepath}"], check=True)
    subprocess.run(["dvc", "push"], check=True)
    print(f"✅ {filepath} добавлен в DVC")