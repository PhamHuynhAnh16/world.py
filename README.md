<div align="center">

# **World.py**

**Trình bao bọc python dành cho world vocoders**

</div>

**Tôi không biết tại sao tôi lại làm cái này:V**

**Bạn có thể sử dụng nó giống như pyworld nhưng tôi không có đưa nó lên pypi**

**Bạn có thể dùng nó như thế này**

```python

import soundfile as sf
import numpy as np

from world import PYWORLD

pw = PYWORLD()
x, fs = sf.read('audio')

f0, t = pw.harvest(x.astype(np.double),  fs=16000, f0_ceil=1100, f0_floor=50, frame_period=10)
f0 = pw.stonemask(x.astype(np.double), 16000, t, f0)

```

**Được viết dựa trên [World](https://github.com/mmorise/World) Của Dr. Morise's**