from __future__ import absolute_import

import numpy as np
import soundfile as sf

def read_audio(file_path: str, 
             dtype: str = 'int16',
             samplerate: int = 16000,
             channels: int = 1):
    assert dtype in ['int16', 'int32', 'float32']
    
    x, _ = sf.read(file_path, dtype=dtype)
    return x