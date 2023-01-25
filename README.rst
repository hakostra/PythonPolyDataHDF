#########################################################
EXPERIMENTAL Python-based PolyData HDF5 reader and writer
#########################################################

A proposition for the vtkPolyData support in the ``vtkhdf`` file format,
implemented in pure Python. Require ``numpy`` and ``h5py`` to be available.

The file ``PythonPolyDataHDF.py`` can be used in two ways:

1.  A Paraview plugin - will provide both reader and writer

2.  A Python module that ban be imported and used
    (``import PythonPolyDataHDF``)


To convert VTP files stored as time series to this format::

    import os

    import vtk
    from PythonPolyDataHDF import PythonPolyDataHDFWriter

    input_path = "/path/to/my/data"

    writer = PythonPolyDataHDFWriter()
    writer.SetFileName("my-output.hdf")
    writer.SetStaticGeometry(1)

    i = 0
    while os.path.isfile(f"{input_path}/pload-{i}.vtp"):
        print(f"Processing step {i}")

        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(f"{input_path}/pload-{i}.vtp")

        reader.UpdateInformation()
        reader.GetPointDataArraySelection().EnableAllArrays()
        reader.GetCellDataArraySelection().EnableAllArrays()

        # First timestep read all arrays - subsequent timesteps disable certain
        # static arrays
        if i > 0:
            reader.GetPointDataArraySelection().DisableArray("area")
            reader.GetPointDataArraySelection().DisableArray("bodyid")
            reader.GetPointDataArraySelection().DisableArray("nvec")

        # Get time
        reader.Update()
        polydata = reader.GetOutput()
        timearr = polydata.GetFieldData().GetAbstractArray("TimeValue")
        time = timearr.GetValue(0)

        writer.SetInputData(polydata)
        writer.WriteNextTime(time)

        i += 1
