ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
ENV["JULIA_PYTHONCALL_EXE"] = joinpath(@__DIR__, "..", "..", ".venv", "bin", "python")  # optional

using PythonCall

P = pyimport("ltc.symbolic.julia")
P.main()