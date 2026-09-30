"""The 0.2.x entry points, kept as thin wrappers over :mod:`hypyrameter.parameters`.

``cubeParamCalculator`` reads an ENVI cube, computes every valid parameter
(or the ones you name), writes a ``<name>_params`` ENVI cube and browse PNGs.
``pointParamCalculator`` does the same for a table of spectra and returns a
DataFrame of parameters by spectrum. New code should call
:func:`hypyrameter.compute` directly.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from hypyrameter import utils as u
from hypyrameter.bands import apply_bad_bands, fill_nans_along_spectrum, to_nanometres
from hypyrameter.browse import valid_browse_products, write_browse_products
from hypyrameter.io import read_envi, write_envi
from hypyrameter.parameters import clamp_reflectance, compute, valid_parameters

logger = logging.getLogger(__name__)


def _ask_for_file() -> str:
    from tkinter import Tk, filedialog  # only when no in_file was given

    root = Tk()
    root.withdraw()
    try:
        return filedialog.askopenfilename(filetypes=[("ENVI header", "*.hdr")])
    finally:
        root.destroy()


def _ask_for_directory() -> str:
    from tkinter import Tk, filedialog

    root = Tk()
    root.withdraw()
    try:
        return filedialog.askdirectory()
    finally:
        root.destroy()


class cubeParamCalculator:
    """Spectral parameters and browse products for an ENVI image cube.

    Args:
        in_file: path to the ``.hdr``; a file dialog asks for one if omitted.
        outdir: where ``run()`` writes; a dialog asks if omitted.
        crop: ``[r0, r1, c0, c1]`` rows and columns to keep.
        bbl: bad-band list (1 = good, 0 = bad); bad bands become NaN.
        interpNans: fill bad bands by linear interpolation along the spectrum.
        flip / transpose: reorient the cube before anything else.
        parameters: names to compute; every valid parameter by default.
        denoise / preview: kept for compatibility; the denoiser lives in
            :mod:`hypyrameter.iovf_generic` and previews are left to the caller.
    """

    def __init__(
        self,
        in_file: str | Path | None = None,
        crop: Sequence[int] | None = None,
        bbl: Sequence[int] | None = None,
        interpNans: bool = False,
        flip: bool = False,
        transpose: bool = False,
        denoise: bool = False,
        preview: bool = False,
        outdir: str | Path | None = None,
        parameters: Sequence[str] | None = None,
    ) -> None:
        if bbl is not None and len(bbl) and bbl[0] is None:  # the old default, [None]
            bbl = None
        self.file = str(in_file) if in_file is not None else _ask_for_file()
        if not self.file:
            raise ValueError("no input file selected")
        self.outdir = str(outdir) if outdir is not None else _ask_for_directory()
        logger.info("loading %s", self.file)
        self.f, self.wvt, self.metadata = read_envi(self.file)
        if self.wvt is None:
            raise ValueError(f"{self.file} has no numeric wavelengths")
        self.f_bands = list(self.wvt)  # nm
        if flip:
            self.f = np.flip(self.f, axis=0)
        if transpose:
            self.f = np.transpose(self.f, (1, 0, 2))
        if crop is not None:
            r0, r1, c0, c1 = crop
            self.f = self.f[r0:r1, c0:c1, :]
        if bbl is not None:
            self.f = apply_bad_bands(self.f, bbl)
            if interpNans:
                self.f = fill_nans_along_spectrum(self.f)
        if denoise:
            raise NotImplementedError(
                "denoise=True is no longer applied here; run hypyrameter.iovf_generic.iovf "
                "on the cube first (needs the 'denoise' extra)"
            )
        self.cube = clamp_reflectance(self.f)
        self.validParams = (
            list(parameters) if parameters is not None else valid_parameters(self.wvt)
        )
        logger.info("valid parameters: %s", self.validParams)
        self.params: np.ndarray | None = None
        if preview:
            self.previewData()

    def previewData(self) -> None:
        """Show the header's default bands (or three reasonable ones) with matplotlib."""
        import matplotlib.pyplot as plt

        if "default bands" in self.metadata:
            bands = [int(float(b)) - 1 for b in self.metadata["default bands"]]
        else:
            bands = [min(i, self.f.shape[-1] - 1) for i in (39, 23, 4)]
        preview = self.f[:, :, bands]
        lo, hi = np.nanmin(preview, axis=(0, 1)), np.nanmax(preview, axis=(0, 1))
        plt.title("Image Preview")
        plt.imshow(np.clip((preview - lo) / (hi - lo), 0, 1))
        plt.show()

    def determineValidParams(self) -> list[str]:
        return valid_parameters(self.wvt)

    def calculateParams(self) -> np.ndarray:
        """(rows, cols, len(validParams)) float32 parameter cube; also kept as ``params``."""
        self.params = compute(
            self.cube,
            self.wvt,
            self.validParams,
            progress=lambda fraction, name: logger.info("%3.0f%% %s", 100 * fraction, name),
        )
        return self.params

    def _parameter_layers(self) -> dict[str, np.ndarray]:
        if self.params is None:
            self.calculateParams()
        return {name: self.params[..., i] for i, name in enumerate(self.validParams)}

    def calculateBrowse(
        self, stype: str = "mad", perc: float = 2, factor: float = 2.5
    ) -> list[str]:
        """Write a PNG per valid browse product into ``outdir``; returns their names."""
        layers = self._parameter_layers()
        self.validBrowseProducts = valid_browse_products(layers)
        write_browse_products(
            layers, self.outdir, self.validBrowseProducts, kind=stype, percent=perc, factor=factor
        )
        return self.validBrowseProducts

    def saveParamCube(self, force: bool = True) -> Path:
        """Write ``<input name>_params.hdr/.img`` into ``outdir``."""
        if self.params is None:
            self.calculateParams()
        name = Path(self.file).stem + "_params.hdr"
        return write_envi(
            Path(self.outdir) / name,
            self.params,
            self.validParams,
            base_metadata=self.metadata,
            default_bands=["R637", "R550", "R463"],
            force=force,
        )

    def run(self) -> np.ndarray:
        """Compute, save the parameter cube and write the browse products."""
        Path(self.outdir).mkdir(parents=True, exist_ok=True)
        self.calculateParams()
        self.saveParamCube()
        self.browseProducts = self.calculateBrowse()
        return self.params


class pointParamCalculator:
    """Spectral parameters for a table of spectra.

    Give either ``data_path`` (a ``.csv`` or a glob of ``.sed`` files) or a
    DataFrame whose first column is wavelength and whose other columns are
    spectra. ``run()`` returns a DataFrame of parameters (rows) by spectrum
    (columns).
    """

    def __init__(
        self,
        data_path: str | None = None,
        df: pd.DataFrame | None = None,
        parameters: Sequence[str] | None = None,
    ) -> None:
        if data_path:
            if data_path.endswith(".csv"):
                df = pd.read_csv(data_path)
            elif data_path.endswith(".sed"):
                df = u.getSedFiles(data_path)
            else:
                raise ValueError("data_path must be a .sed or .csv file")
        if df is None or df is False:
            raise ValueError("give data_path or df")
        self.wvt = to_nanometres(df.iloc[:, 0].to_numpy(dtype=np.float64))
        self.spectra = df.iloc[:, 1:]
        self.specNames = list(self.spectra.columns)
        self.validParams = (
            list(parameters) if parameters is not None else valid_parameters(self.wvt)
        )
        logger.info("valid parameters: %s", self.validParams)

    def determineValidParams(self) -> list[str]:
        return valid_parameters(self.wvt)

    def run(self) -> pd.DataFrame:
        table = self.spectra.to_numpy(dtype=np.float64).T  # (spectra, bands)
        values = compute(table, self.wvt, self.validParams, dtype=np.float64)  # (spectra, params)
        self.parameter_df = pd.DataFrame(
            values.T, index=pd.Index(self.validParams), columns=self.specNames
        )
        return self.parameter_df
