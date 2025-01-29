import os, torch, ctypes, platform, shutil, numpy as np, tempfile as tf

class DioOption(ctypes.Structure):
    _fields_ = [("F0Floor", ctypes.c_double), ("F0Ceil", ctypes.c_double), ("ChannelsInOctave", ctypes.c_double), ("FramePeriod", ctypes.c_double), ("Speed", ctypes.c_int), ("AllowedRange", ctypes.c_double)]

class HarvestOption(ctypes.Structure):
    _fields_ = [("F0Floor", ctypes.c_double), ("F0Ceil", ctypes.c_double), ("FramePeriod", ctypes.c_double)]

class CheapTrickOption(ctypes.Structure):
    _fields_ = [("Q1", ctypes.c_double), ("F0Floor", ctypes.c_double), ("FftSize", ctypes.c_int)]

class D4COption(ctypes.Structure):
    _fields_ = [("Threshold", ctypes.c_double)]

class PYWORLD:
    def __init__(self):
        model = torch.load("world.pth", map_location="cpu")
        model_type, suffix = (("world_64" if platform.architecture()[0] == "64bit" else "world_86"), ".dll") if platform.system() == "Windows" else ("world_linux", ".so")

        temp_folder = "temp"

        if os.path.exists(temp_folder): shutil.rmtree(temp_folder, ignore_errors=True)
        os.makedirs(temp_folder, exist_ok=True)

        with tf.NamedTemporaryFile(delete=False, suffix=suffix, dir=temp_folder) as tmp:
            tmp.write(model[model_type])
            temp_path = tmp.name

        self.world_dll = ctypes.CDLL(temp_path)

    def harvest(self, x, fs, f0_floor=50, f0_ceil=1100, frame_period=10):
        self.world_dll.Harvest.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int, ctypes.POINTER(HarvestOption), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
        self.world_dll.Harvest.restype = None 
        self.world_dll.InitializeHarvestOption.argtypes = [ctypes.POINTER(HarvestOption)]
        self.world_dll.InitializeHarvestOption.restype = None
        self.world_dll.GetSamplesForHarvest.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_double]
        self.world_dll.GetSamplesForHarvest.restype = ctypes.c_int
        option = HarvestOption()
        self.world_dll.InitializeHarvestOption(ctypes.byref(option))
        option.F0Floor = f0_floor
        option.F0Ceil = f0_ceil
        option.FramePeriod = frame_period
        f0_length = self.world_dll.GetSamplesForHarvest(fs, len(x), option.FramePeriod)
        f0 = (ctypes.c_double * f0_length)()
        tpos = (ctypes.c_double * f0_length)()
        self.world_dll.Harvest((ctypes.c_double * len(x))(*x), len(x), fs, ctypes.byref(option), tpos, f0)
        return np.array(f0, dtype=np.float64), np.array(tpos, dtype=np.float64)

    def dio(self, x, fs, f0_floor=50, f0_ceil=1100, channels_in_octave=2, frame_period=10, speed=1, allowed_range=0.1):
        self.world_dll.Dio.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int, ctypes.POINTER(DioOption), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
        self.world_dll.Dio.restype = None  
        self.world_dll.InitializeDioOption.argtypes = [ctypes.POINTER(DioOption)]
        self.world_dll.InitializeDioOption.restype = None
        self.world_dll.GetSamplesForDIO.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_double]
        self.world_dll.GetSamplesForDIO.restype = ctypes.c_int
        option = DioOption()
        self.world_dll.InitializeDioOption(ctypes.byref(option))
        option.F0Floor = f0_floor
        option.F0Ceil = f0_ceil
        option.ChannelsInOctave = channels_in_octave
        option.FramePeriod = frame_period
        option.Speed = speed
        option.AllowedRange = allowed_range
        f0_length = self.world_dll.GetSamplesForDIO(fs, len(x), option.FramePeriod)
        f0 = (ctypes.c_double * f0_length)()
        tpos = (ctypes.c_double * f0_length)()
        self.world_dll.Dio((ctypes.c_double * len(x))(*x), len(x), fs, ctypes.byref(option), tpos, f0)
        return np.array(f0, dtype=np.float64), np.array(tpos, dtype=np.float64)

    def stonemask(self, x, fs, tpos, f0):
        self.world_dll.StoneMask.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
        self.world_dll.StoneMask.restype = None 
        out_f0 = (ctypes.c_double * len(f0))()
        self.world_dll.StoneMask((ctypes.c_double * len(x))(*x), len(x), fs, (ctypes.c_double * len(tpos))(*tpos), (ctypes.c_double * len(f0))(*f0), len(f0), out_f0)
        return np.array(out_f0, dtype=np.float64)

    def initialize_option(self, fs):
        self.world_dll.InitializeCheapTrickOption.argtypes = [ctypes.c_int, ctypes.POINTER(CheapTrickOption)]
        self.world_dll.InitializeCheapTrickOption.restype = None
        option = CheapTrickOption()
        self.world_dll.InitializeCheapTrickOption(fs, ctypes.byref(option))
        return option

    def get_fft_size(self, fs, f0_floor=71.0):
        self.world_dll.GetFFTSizeForCheapTrick.argtypes = [ctypes.c_int, ctypes.POINTER(CheapTrickOption)]
        self.world_dll.GetFFTSizeForCheapTrick.restype = ctypes.c_int
        option = self.initialize_option(fs)
        option.F0Floor = f0_floor
        return int(self.world_dll.GetFFTSizeForCheapTrick(fs, ctypes.byref(option)))

    def cheaptrick(self, x, fs, t, f0, q1=-0.15, f0_floor=71.0, fft_size=None):
        self.world_dll.CheapTrick.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.POINTER(CheapTrickOption), ctypes.POINTER(ctypes.POINTER(ctypes.c_double))]
        self.world_dll.CheapTrick.restype = None
        option = self.initialize_option(fs)
        option.Q1 = q1
        option.F0Floor = f0_floor
        option.FftSize = fft_size or self.get_fft_size(fs, f0_floor)
        fft_size_half = option.FftSize // 2 + 1
        spectrogram = np.zeros((len(f0), fft_size_half), dtype=np.float64)
        spectrogram_pointers = (ctypes.POINTER(ctypes.c_double) * len(f0))()
        for i in range(len(f0)):
            spectrogram_pointers[i] = spectrogram[i].ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        self.world_dll.CheapTrick(x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(x), fs, t.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), f0.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(f0), ctypes.byref(option), spectrogram_pointers)
        return np.array(spectrogram, dtype=np.float64)

    def get_f0_floor(self, fs, fft_size):
        self.world_dll.GetF0FloorForCheapTrick.argtypes = [ctypes.c_int, ctypes.c_int]
        self.world_dll.GetF0FloorForCheapTrick.restype = ctypes.c_double
        return float(self.world_dll.GetF0FloorForCheapTrick(fs, fft_size))

    def get_number_of_aperiodicities(self, fs):
        self.world_dll.GetNumberOfAperiodicities.argtypes = [ctypes.c_int]
        self.world_dll.GetNumberOfAperiodicities.restype = ctypes.c_int
        return int(self.world_dll.GetNumberOfAperiodicities(fs))

    def code_aperiodicity(self, ap, fs):
        self.world_dll.CodeAperiodicity.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_double)), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.POINTER(ctypes.c_double))]
        self.world_dll.CodeAperiodicity.restype = None
        bap = np.zeros((ap.shape[0], self.get_number_of_aperiodicities(fs)), dtype=np.float64)
        ap_ptrs = (ap.__array_interface__['data'][0] + np.arange(ap.shape[0]) * ap.strides[0]).astype(np.uintp)
        bap_ptrs = (bap.__array_interface__['data'][0] + np.arange(bap.shape[0]) * bap.strides[0]).astype(np.uintp)
        self.world_dll.CodeAperiodicity(ap_ptrs.ctypes.data_as(ctypes.POINTER(ctypes.POINTER(ctypes.c_double))), ap.shape[0], fs, ((ap.shape[1] - 1) * 2), bap_ptrs.ctypes.data_as(ctypes.POINTER(ctypes.POINTER(ctypes.c_double))))
        return np.array(bap, dtype=np.float64)

    def decode_aperiodicity(self, bap, fs, fft_size):
        self.world_dll.DecodeAperiodicity.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_double)), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.POINTER(ctypes.c_double))]
        self.world_dll.DecodeAperiodicity.restype = None
        ap = np.zeros((bap.shape[0], fft_size // 2 + 1), dtype=np.float64)
        bap_ptrs = (bap.__array_interface__['data'][0] + np.arange(bap.shape[0]) * bap.strides[0]).astype(np.uintp)
        ap_ptrs = (ap.__array_interface__['data'][0] + np.arange(ap.shape[0]) * ap.strides[0]).astype(np.uintp)
        self.world_dll.DecodeAperiodicity(bap_ptrs.ctypes.data_as(ctypes.POINTER(ctypes.POINTER(ctypes.c_double))), bap.shape[0], fs, fft_size, ap_ptrs.ctypes.data_as(ctypes.POINTER(ctypes.POINTER(ctypes.c_double))))
        return np.array(ap, dtype=np.float64)

    def code_spectral_envelope(self, sp, fs, num_dimensions):
        self.world_dll.CodeSpectralEnvelope.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_double)), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.POINTER(ctypes.c_double))]
        self.world_dll.CodeSpectralEnvelope.restype = None
        mgc = np.zeros((sp.shape[0], num_dimensions), dtype=np.float64)
        sp_ptrs = (sp.__array_interface__['data'][0] + np.arange(sp.shape[0]) * sp.strides[0]).astype(np.uintp)
        mgc_ptrs = (mgc.__array_interface__['data'][0] + np.arange(mgc.shape[0]) * mgc.strides[0]).astype(np.uintp)
        self.world_dll.CodeSpectralEnvelope(sp_ptrs.ctypes.data_as(ctypes.POINTER(ctypes.POINTER(ctypes.c_double))), sp.shape[0], fs, ((sp.shape[1] - 1) * 2), num_dimensions, mgc_ptrs.ctypes.data_as(ctypes.POINTER(ctypes.POINTER(ctypes.c_double))))
        return np.array(mgc, dtype=np.float64)

    def decode_spectral_envelope(self, mgc, fs, fft_size):
        self.world_dll.DecodeSpectralEnvelope.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_double)), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.POINTER(ctypes.c_double))]
        self.world_dll.DecodeSpectralEnvelope.restype = None
        sp = np.zeros((mgc.shape[0], fft_size // 2 + 1), dtype=np.float64)
        mgc_ptrs = (mgc.__array_interface__['data'][0] + np.arange(mgc.shape[0]) * mgc.strides[0]).astype(np.uintp)
        sp_ptrs = (sp.__array_interface__['data'][0] + np.arange(sp.shape[0]) * sp.strides[0]).astype(np.uintp)
        self.world_dll.DecodeSpectralEnvelope(mgc_ptrs.ctypes.data_as(ctypes.POINTER(ctypes.POINTER(ctypes.c_double))), mgc.shape[0], fs, fft_size, mgc.shape[1], sp_ptrs.ctypes.data_as(ctypes.POINTER(ctypes.POINTER(ctypes.c_double))))
        return np.array(sp, dtype=np.float64)

    def d4c(self, x, fs, tpos, f0, threshold=0.85, fft_size=None):
        self.world_dll.D4C.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int, ctypes.POINTER(D4COption), ctypes.POINTER(ctypes.POINTER(ctypes.c_double))]
        self.world_dll.D4C.restype = None
        self.world_dll.InitializeD4COption.argtypes = [ctypes.POINTER(D4COption)]
        self.world_dll.InitializeD4COption.restype = None
        if fft_size is None: fft_size = self.get_fft_size(fs) 
        option = D4COption()
        self.world_dll.InitializeD4COption(ctypes.byref(option))
        option.Threshold = threshold
        ap = np.zeros((len(f0), fft_size // 2 + 1), dtype=np.float64)
        ap_ptrs = (ap.__array_interface__['data'][0] + np.arange(ap.shape[0]) * ap.strides[0]).astype(np.uintp)
        self.world_dll.D4C(x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(x), fs, tpos.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), f0.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(f0), fft_size, ctypes.byref(option), ap_ptrs.ctypes.data_as(ctypes.POINTER(ctypes.POINTER(ctypes.c_double))))
        return np.array(ap, dtype=np.float64)

    def synthesis(self, f0, sp, ap, fs, frame_period=5.0):
        self.world_dll.Synthesis.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.POINTER(ctypes.POINTER(ctypes.c_double)), ctypes.POINTER(ctypes.POINTER(ctypes.c_double)), ctypes.c_int, ctypes.c_double, ctypes.c_int, ctypes.c_int, np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")]
        self.world_dll.Synthesis.restype = None
        f0 = np.ascontiguousarray(f0, dtype=np.float64)
        sp = np.ascontiguousarray(sp, dtype=np.float64)
        ap = np.ascontiguousarray(ap, dtype=np.float64)
        y_length = int(len(f0) * frame_period * fs / 1000)
        y = np.zeros(y_length, dtype=np.float64)
        self.world_dll.Synthesis(f0, len(f0), (ctypes.POINTER(ctypes.c_double) * sp.shape[0])(*[sp[row, :].ctypes.data_as(ctypes.POINTER(ctypes.c_double)) for row in range(sp.shape[0])]), (ctypes.POINTER(ctypes.c_double) * ap.shape[0])(*[ap[row, :].ctypes.data_as(ctypes.POINTER(ctypes.c_double)) for row in range(ap.shape[0])]), ((sp.shape[1] - 1) * 2), frame_period, fs, y_length, y)
        return np.array(y, dtype=np.float64)

    def wav2world(self, x, fs, fft_size=None, frame_period=5.0):
        _f0, t = self.dio(x=x, fs=fs, frame_period=frame_period)
        f0 = self.stonemask(x=x, f0=_f0, tpos=t, fs=fs)
        return f0, self.cheaptrick(x=x, f0=f0, t=t, fs=fs, fft_size=fft_size), self.d4c(x=x, f0=f0, tpos=t, fs=fs, fft_size=fft_size)