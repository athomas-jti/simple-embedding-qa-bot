import numpy as np
from torch import Tensor

class VectorIndexResult:
    def __init__(self, value, vector, distance):
        self.value = value
        self.vector = vector
        self.distance = distance


class ExactVectorIndex:
    def __init__(self, vect_from_string=None):
        self._keys = None
        self._values = []
        self._vect_from_string = vect_from_string
    def __setitem__(self, key, value):
        if value is None:
            pass
        if isinstance(key, str):
            if self._vect_from_string:
                key = self._vect_from_string(key)
            else:
                raise TypeError('for `key` wanted a vector type, got str')
        if isinstance(key, Tensor):
            key = key.detach()
        count = len(self._values)
        if self._keys is None:
            dimension = key.shape[0]
            self._keys = np.empty((2,dimension))
        if count >= self._keys.shape[0]:
            dimension = self._keys.shape[1]
            self._keys = np.resize(self._keys, (count*2,dimension))        
        self._keys[count, :] = key
        self._values.append(value)
    def __getitem__(self, key):
        if isinstance(key, str):
            if self._vect_from_string:
                key = self._vect_from_string(key)
            else:
                raise TypeError('for `key` wanted a vector type, got str')
        if isinstance(key, Tensor):
            key = key.detach().numpy()
        if not self._values:
            raise KeyError('<EMPTY>')
        ix = np.argmin(np.linalg.norm(self._keys[:len(self),:] - key, axis=1))
        return self._values[ix]
    def __len__(self):
        return len(self._values)
    def get_k_nearest(self, key, k):
        if isinstance(key, str):
            if self._vect_from_string:
                key = self._vect_from_string(key)
            else:
                raise TypeError('for `key` wanted a vector type, got str')
        if isinstance(key, Tensor):
            key = key.detach().numpy()
        if not self._values:
            raise KeyError('<EMPTY>')
        distances = np.linalg.norm(self._keys[:len(self),:] - key, axis=1)
        items = [VectorIndexResult(
            value=self._values[i],
            vector=self._keys[i,:],
            distance=distances[i]
            ) for i in range(len(self))]
        items.sort(key=lambda x:x.distance)
        return items[:k]
