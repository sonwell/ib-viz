import sys
import enum
import struct
import numpy as np
from scipy.sparse import csr_matrix


class Layout(enum.Enum):
    DENSE = b'd'
    SPARSE = b's'


class Container(enum.Enum):
    VECTOR = b've'
    MATRIX = b'ma'



class BinaryOpener:
    def __init__(self, filename, mode='rb'):
        self.fd = self._get_buffer(filename, mode)

    def _get_buffer(self, filename, mode):
        if filename == '-':
            if 'r' in mode:
                return sys.stdin.buffer
            return sys.stdout.buffer
        if 'b' not in mode:
            mode += 'b'
        return open(filename, mode)


class Reader(BinaryOpener):
    def _read_magic(self):
        TYPES = {b'd': ('d', 8, np.double),
                 b's': ('f', 4, np.single),
                 b'c': ('ff', 8, np.csingle),
                 b'z': ('dd', 16, np.cdouble)}
        type, layout, *container = self.unpack('c', 1, 4)
        return TYPES.get(type), Layout(layout), \
            Container(b''.join(container))

    def unpack(self, format, size, n):
        buf = self.fd.read(size * n)
        if not buf:
            raise IOError
        return struct.unpack(format * n, buf)

    def _format_values(self, format, dtype, values):
        if len(format) == 1:
            return np.array(values, dtype=dtype)
        n = len(values)
        assert n & 1 == 0
        values = np.array(values, dtype=dtype).reshape((n//2, 2))
        return (values @ np.array([[1], [1j]], dtype=dtype)).flatten()

    def _parse_sparse_vector(self, type_info, rows, cols, nnz):
        assert cols == 1
        format, size, dtype = type_info
        values = self.unpack(format, size, nnz)
        indices = self.unpack('i', 4, nnz)
        values = self._format_values(format, dtype, values)
        return csc_matrix((values, indices, [0, nnz]), shape=(rows, cols))

    def _parse_sparse_matrix(self, type_info, rows, cols, nnz):
        format, size, dtype = type_info
        values = self.unpack(format, size, nnz)
        indices = self.unpack('i', 4, nnz)
        starts = self.unpack('i', 4, rows+1)
        values = self._format_values(format, dtype, values)
        return csr_matrix((values, indices, starts), shape=(rows, cols))

    def _parse_dense(self, type_info, rows, cols, nnz):
        format, size, dtype = type_info
        values = self.unpack(format, size, nnz)
        values = self._format_values(format, dtype, values)
        return np.array(values).reshape((rows, cols), order='F')

    @classmethod
    def _get_method(cls, layout, container):
        if layout == Layout.DENSE:
            return cls._parse_dense
        elif layout == Layout.SPARSE:
            if container == Container.VECTOR:
                return cls._parse_sparse_vector
            if container == Contianer.MATRIX:
                return cls._parse_sparse_matrix
        raise ValueError("Unexpected values in bytestream: " \
                         f"{layout}{container}")

    def _parse(self, type, layout, container):
        rows, cols, nnz = self.unpack('i', 4, 3)
        method = self._get_method(layout, container)
        return method(self, type, rows, cols, nnz)

    def parse(self):
        type, layout, container = self._read_magic()
        return self._parse(type, layout, container)

    def read(self, _):
        raise NotImplementedError(f"{self.__class__.__name__}" \
                                  "should not be instantiated directly")

    def __iter__(self):
        t = None
        while True:
            try:
                s, *others = self.read()
                if t is None or s > t:
                    yield s, *others
                t = s
            except (IOError, OSError) as e:
                break


class Writer(BinaryOpener):
    def _guess_container(self, x, force):
        if force is not None:
            if isinstance(force, Container):
                return force
            if isinstance(force, str):
                force = force.lower().encode('utf-8')
            if isinstance(force, bytes):
                return Container(force[:2])
        if len(x.shape) == 1 or x.shape[1] == 1:
            return Container.VECTOR
        return Container.MATRIX

    def pack(self, fmt, *x):
        return self.fd.write(struct.pack(fmt, *x))

    def _write_sizes(self, rows, cols, nnz):
        return self.pack('iii', rows, cols, nnz)

    def _serialize_sparse_matrix(self, x, fmt):
        pass

    def _serialize_sparse_vector(self, x, fmt):
        pass

    def _serialize_dense(self, x, fmt):
        rows, cols, *_ = x.shape + (1,)
        nnz = rows * cols
        self._write_sizes(rows, cols, nnz)
        data = x.flatten(order='F').tolist()
        if len(fmt) > 1:  # complex types
            data = np.array([np.real(data), np.imag(data)]).flatten()
        return self.pack(fmt * nnz, *data)

    def _serialize(self, x, container, layout, fmt, type):
        container = container.value
        self.pack('cccc', layout.value, type, container[0:1], container[1:2])
        method = self._serialize_dense
        if layout == Layout.SPARSE:
            if container == Container.MATRIX:
                method = self._serialize_sparse_matrix
            else:
                method = self._serialize_sparse_vector
        return method(x, fmt)

    def serialize(self, x, container=None):
        if isinstance(x, float):  # times
            return self.pack('d', x)
        container = self._guess_container(x, container)
        layout = Layout.SPARSE if isinstance(x, csr_matrix) else Layout.DENSE
        if x.dtype == np.single:
            fmt, type = 's', b's'
        elif x.dtype == np.double:
            fmt, type = 'd', b'd'
        elif x.dtype == np.csingle:
            fmt, type = 'ss', b'c'
        elif x.dtype == np.cdouble:
            fmt, type = 'dd', b'z'
        return self._serialize(x, container, layout, fmt, type)

    def write(self, *args):
        raise NotImplementedError(f"{self.__class__.__name__}" \
                                  "should not be instantiated directly")


class FluidParser(Reader, Writer):
    def read(self):
        time, = self.unpack('d', 8, 1)
        velocity = self.parse(), self.parse(), self.parse()
        pressure = self.parse()
        return time, velocity, pressure

    def write(self, time, velocity, pressure):
        self.pack('d', time)
        self.serialize(velocity[0], Container.VECTOR)
        self.serialize(velocity[1], Container.VECTOR)
        self.serialize(velocity[2], Container.VECTOR)
        self.serialize(pressure, Container.VECTOR)


class SimulationState(FluidParser):
    def __init__(self, n, filename, mode='rb'):
        "n: number of IB objects"
        self.n = n
        super().__init__(self, filename, mode=mode)

    def read(self):
        time, velocity, pressure = super().read()
        others = tuple(self.parse() for _ in range(self.n))
        return time, velocity, pressure, *others

    def write(self, time, velocity, pressure, *args):
        super().write(time, velocity, pressure)
        for arg in args:
            self.serialize(arg, Container.MATRIX)


class InitialParser(SimulationState):
    def __init__(self, filename, mode='rb'):
        super().__init__(self, 2, filename, mode=mode)


class WholeParser(SimulationState):
    def __init__(self, filename, mode='rb'):
        super().__init__(self, 3, filename, mode=mode)


if __name__ == '__main__':
    parser = WholeParser('flat.bin')
    for frame in parser:
        time, velocity, pressure, rbc, plt, endo = frame
        print(time)

