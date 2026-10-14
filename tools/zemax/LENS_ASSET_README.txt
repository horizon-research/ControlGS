Google Cardboard RGB Zemax inputs

Contents: google_cardboard_rgb.ZMX (lens design), .zar (archive), .ZDA (companion data), psf_extraction.py, config_example.json, and lens-assets-manifest.json (SHA-256 checksums).

To extract geometric PSFs on Windows with Zemax OpticStudio:
  python -m pip install zospy numpy matplotlib tqdm
  cd tools/zemax
  python psf_extraction.py --config config_example.json

The example writes NPZ, NPY and preview PNG files under tools/zemax/psf_geometric_scan/. It samples a +/-12 degree field with 30 boundary samples per axis, three RGB wavelengths and 64x64 PSFs.

Training and inference use the separately released precomputed PSF, downloaded automatically, and do not require Zemax. The original PSF exports do not record a source-lens hash or complete extraction history; treat a regenerated PSF as a new optical asset version.
