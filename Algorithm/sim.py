import ctypes
from ctypes import c_float, c_bool, c_int

MosArray = (c_float * 39) * 50
RArray = (c_float * 18) * 50
CArray = (c_float * 18) * 50
VArray = (c_float * 18) * 1
IArray = (c_float * 18) * 1
ParmArray = c_float * 20

class Performance(ctypes.Structure):
    _fields_ = [
        ("gain",           c_float),
        ("gbw",            c_float),
        ("phase_margin",   c_float),
        ("idc",            c_float),
        ("noise",          c_float),
        ("cmrr",           c_float),
        ("sr",             c_float),
        ("psrr",           c_float),
        ("error",          c_bool),
        ("mos",            MosArray),
        ("R",              RArray),
        ("C",              CArray),
        ("V",              VArray),
        ("I",              IArray),
    ]

class Param(ctypes.Structure):
    _fields_ = [
        ("dev_parm",       ParmArray),
        ("sim_times",      c_int),
        ("id",             c_int),
    ]
    def __init__(self, dev_list, sim_times, id_val):
        super().__init__()
        self.dev_parm[:] = dev_list
        self.sim_times = sim_times
        self.id = id_val


if __name__ == "__main__":
    params = Param([1, 25, 1, 25, 1, 25, 1, 25, 1, 25, 1, 10, 1, 20, 1, 20, 1, 20, 5, 62300],1,ord('a'))

    lib = ctypes.CDLL("./simulation.so")
    print("start")

    #lib.Simulate.argtypes = [Param]
    #lib.Simulate(params)
    lib.Get_Op_Range.argtypes = [Param]
    print("mid")
    lib.Get_Op_Range(params)
    print("end")

    final_perf = Performance.in_dll(lib,"final_perf")
    print(final_perf.mos[2][3])