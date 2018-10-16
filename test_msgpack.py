import msgpack, json
import numpy as np

x = np.random.randn(500).tolist()

with open('data.dat', 'wb') as f:
    msgpack.dump(x, f)

with open('data.json', 'w') as f:
    json.dump(x, f)

