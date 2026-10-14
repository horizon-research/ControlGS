"""Download and restore the released scene assets from public Google Drive links."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SCENES = {"stump": "mip360", "materials": "nerf_synthetic"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def matches(path, entry):
    return path.is_file() and path.stat().st_size == entry["bytes"] and sha256(path) == entry["sha256"]


def runtime_entries(scene, root=ROOT):
    prefix = "models/" + SCENES[scene] + "/" + scene + "/"
    models = json.loads((root / "checkpoints/manifest.json").read_text())["files"]
    optics = json.loads((root / "assets-manifest.json").read_text())["files"]
    return [entry for entry in models if entry["path"].startswith(prefix)], optics


def verify_scene(scene, root=ROOT):
    groups = runtime_entries(scene, root)
    for entry in groups[0] + groups[1]:
        if not matches(root / entry["path"], entry):
            raise RuntimeError("Missing or invalid asset: " + entry["path"])
    print(f"Verified {scene}: {len(groups[0])} checkpoint files and {len(groups[1])} PSF resources", flush=True)


def install_bundle(archive, entry, root=ROOT):
    if not matches(archive, entry):
        raise RuntimeError("Asset archive checksum mismatch: " + str(archive))
    allowed = ("models/", "optics/psf_geometric_scan/")
    with tarfile.open(archive, "r:gz") as source:
        members = source.getmembers()
        for member in members:
            name = Path(member.name)
            if not member.isfile() or name.is_absolute() or ".." in name.parts or not member.name.startswith(allowed):
                raise RuntimeError("Unexpected archive entry: " + member.name)
            target = root / name
            if not target.resolve().is_relative_to(root.resolve()):
                raise RuntimeError("Archive entry escapes the repository: " + member.name)
        for member in members:
            target = root / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.extractfile(member) as incoming:
                if target.exists():
                    digest = hashlib.sha256()
                    for block in iter(lambda: incoming.read(1024 * 1024), b""):
                        digest.update(block)
                    if not target.is_file() or sha256(target) != digest.hexdigest():
                        raise RuntimeError("Existing asset differs; move it aside before restoring: " + str(target))
                    continue
                with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as staging:
                    temporary = Path(staging.name)
                    try:
                        shutil.copyfileobj(incoming, staging, 1024 * 1024)
                    except BaseException:
                        temporary.unlink(missing_ok=True)
                        raise
                temporary.chmod(member.mode & 0o777)
                os.replace(temporary, target)


def download_bundle(entry, cache):
    archive = cache / entry["file"]
    if matches(archive, entry):
        print("Using cached archive: " + archive.name, flush=True)
        return archive
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError("Install the repository requirements to enable Google Drive downloads") from exc
    file_id = entry.get("google_drive_id")
    if not file_id:
        raise RuntimeError("No published download is configured for " + entry["file"])
    cache.mkdir(parents=True, exist_ok=True)
    partial = archive.with_name(archive.name + ".part")
    print("Downloading " + archive.name + " from Google Drive", flush=True)
    downloaded = gdown.download(id=file_id, output=str(partial), quiet=False, use_cookies=False, resume=True)
    if downloaded is None:
        raise RuntimeError("Download failed: " + entry["file"] + "; obtain the archive from its published link or retry")
    if not matches(partial, entry):
        partial.unlink(missing_ok=True)
        raise RuntimeError("Download failed or checksum differs: " + entry["file"] + "; retry or obtain the archive from its published link")
    os.replace(partial, archive)
    return archive


def ensure_assets(scene, *, root=ROOT, cache=None):
    groups = runtime_entries(scene, root)
    bundles = json.loads((root / "checkpoints/downloads.json").read_text())["bundles"]
    cache = Path(cache).expanduser().resolve() if cache else root / ".cache/assets"
    for key, entries in [(scene, groups[0]), ("optics", groups[1])]:
        if all(matches(root / entry["path"], entry) for entry in entries):
            continue
        entry = bundles[key]
        archive = download_bundle(entry, cache)
        install_bundle(archive, entry, root)
    verify_scene(scene, root)
