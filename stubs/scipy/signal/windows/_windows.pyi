from _typeshed import Incomplete
import numpy.typing as npt


def general_cosine(M, a, sym: bool = ...): ...
def boxcar(M, sym: bool = ...): ...
def triang(M, sym: bool = ...): ...
def parzen(M, sym: bool = ...): ...
def bohman(M, sym: bool = ...): ...
def blackman(M, sym: bool = ...): ...
def nuttall(M, sym: bool = ...): ...
def blackmanharris(M, sym: bool = ...): ...
def flattop(M, sym: bool = ...): ...
def bartlett(M, sym: bool = ...): ...
def hann(M, sym: bool = ...): ...
def tukey(M, alpha: float = ..., sym: bool = ...): ...
def barthann(M, sym: bool = ...): ...
def general_hamming(M, alpha, sym: bool = ...): ...
def hamming(M, sym: bool = ...): ...
def kaiser(M, beta, sym: bool = ...): ...
def kaiser_bessel_derived(M, beta, *, sym: bool = ...): ...
def gaussian(M, std, sym: bool = ...): ...
def general_gaussian(M, p, sig, sym: bool = ...): ...
def chebwin(M, at, sym: bool = ...): ...
def cosine(M, sym: bool = ...): ...


def exponential(M, center: Incomplete | None = ...,
                tau: float = ..., sym: bool = ...): ...


def taylor(M, nbar: int = ..., sll: int = ...,
           norm: bool = ..., sym: bool = ...): ...
def dpss(M, NW, Kmax: Incomplete | None = ..., sym: bool = ...,
         norm: Incomplete | None = ..., return_ratios: bool = ...): ...


def lanczos(M, *, sym: bool = ...): ...
def get_window(window: str | float | tuple, Nx: int, fftbins: bool = ...) -> npt.NDArray: ...
