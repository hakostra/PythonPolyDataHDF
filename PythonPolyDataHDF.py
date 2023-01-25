#!/usr/bin/env python3

# Python standard library imports
from pathlib import Path

# Third party packages
import h5py as h5
import numpy as np
import vtk
from vtk.util import numpy_support as nps
from vtk.util.vtkAlgorithm import VTKPythonAlgorithmBase


__version__ = "1.0"


class PythonPolyDataHDFBase(VTKPythonAlgorithmBase):
    # One place to define certain properties for Paraview decorators
    extensions = "hdf vtkhdf",
    file_description = "VTKHDF file format",

    def __init__(self):
        # HDF5 cache parameters - None means use h5py defaults
        self.rdcc_nbytes = None
        self.rdcc_w0 = None
        self.rdcc_nslots = None

        # Filename to read
        self.filename = None

        # HDF5 file handle
        self.fh = None

        self.time_values = None
        self.time_index = 0
        self.ntimes = 0

    def __del__(self):
        if self.fh is not None:
            self.fh.close()
        self.fh = None

    def SetFileName(self, filename):
        if isinstance(filename, str):
            if self.filename != filename:
                self.filename = filename
        elif isinstance(filename, Path):
            if self.filename != str(filename):
                self.filename = str(filename)
        else:
            raise RuntimeError(f"Expected str, got {type(filename)}")

        self.Modified()

    def GetFileName(self):
        return self.filename

    def SetRdcc(self, rdcc_nbytes=None, rdcc_w0=None, rdcc_nslots=None):
        """Set the Raw Data Chunk Cache parameters on opening the HDF5 file
        I do not call Modified() here becuase it technically does not
        modify the data provided by the reader"""
        if rdcc_nbytes:
            self.SetRdcc_nbytes(rdcc_nbytes)

        if rdcc_w0:
            self.SetRdcc_w0(rdcc_w0)

        if rdcc_nslots:
            self.SetRdcc_nslots(rdcc_nslots)

    def SetRdcc_nbytes(self, rdcc_nbytes):
        if isinstance(rdcc_nbytes, int):
            self.rdcc_nbytes = rdcc_nbytes
        else:
            raise RuntimeError(f"Expected int, got {type(rdcc_nbytes)}")

    def GetRdcc_nbytes(self):
        return self.rdcc_nbytes

    def SetRdcc_w0(self, rdcc_w0):
        if isinstance(rdcc_w0, float):
            self.rdcc_w0 = rdcc_w0
        else:
            raise RuntimeError(f"Expected float, got {type(rdcc_w0)}")

    def GetRdcc_w0(self):
        return self.rdcc_w0

    def SetRdcc_nslots(self, rdcc_nslots):
        if isinstance(rdcc_nslots, int):
            self.rdcc_nslots = rdcc_nslots
        else:
            raise RuntimeError(f"Expected int, got {type(rdcc_nslots)}")

    def GetRdcc_nslots(self):
        return self.rdcc_nslots

    def open_file(self, mode="r"):
        if self.fh is None:
            if self.filename is None:
                raise RuntimeError

            self.fh = h5.File(self.filename, mode,
                              rdcc_nbytes=self.rdcc_nbytes,
                              rdcc_w0=self.rdcc_w0,
                              rdcc_nslots=self.rdcc_nslots)

        return self.fh


class PythonPolyDataHDFWriter(PythonPolyDataHDFBase):
    version = (2, 0)
    type = "PolyData"

    def __init__(self):
        VTKPythonAlgorithmBase.__init__(self, nInputPorts=1, nOutputPorts=0,
                                        inputType='vtkPolyData')
        super().__init__()

        self.time = None
        self.static_geometry = 1

    def GetStaticGeom(self):
        return self.static_geometry

    def SetStaticGeometry(self, static_geometry):
        if isinstance(static_geometry, int):
            self.static_geometry = static_geometry
        else:
            raise RuntimeError(f"Expected int, got {type(static_geometry)}")

        self.Modified()

    def Write(self):
        self.Modified()
        self.Update()

    def WriteNextTime(self, time):
        self.time = time
        self.ntimes += 1
        if self.time_values is None:
            self.time_values = [time, ]
        else:
            self.time_values.append(time)

        self.Modified()
        self.Update()

    def RequestUpdateExtent(self, request, inInfo, outInfo):
        info = inInfo[0].GetInformationObject(0)
        has_time = info.Has(vtk.vtkStreamingDemandDrivenPipeline.TIME_STEPS())

        if has_time:
            self.time_values = info.Get(
                vtk.vtkStreamingDemandDrivenPipeline.TIME_STEPS())
            self.ntimes = len(self.time_values)

        # If the number of timesteps is bigger than 1, write out all
        if self.ntimes > 1:
            self.time = self.time_values[self.time_index]
            info.Set(vtk.vtkStreamingDemandDrivenPipeline.UPDATE_TIME_STEP(),
                     self.time)

        return 1

    def RequestData(self, request, inInfo, outInfo):
        # First timestep create the HDF5 file - truncate if it exist
        if self.time_index == 0:
            self.create_file()

        # Write data
        self.time = self.time_values[self.time_index]
        polydata = self.GetInputData(inInfo, 0, 0)
        self.write(polydata)

        # Increment time
        self.time_index += 1

        if self.ntimes > 0:
            request.Set(vtk.vtkStreamingDemandDrivenPipeline.
                        CONTINUE_EXECUTING(), 1)

        if self.time_index >= self.ntimes:
            request.Set(vtk.vtkStreamingDemandDrivenPipeline.
                        CONTINUE_EXECUTING(), 0)

        return 1

    def SetInputData(self, data):
        return self.SetInputDataObject(0, data)

    def create_file(self):
        """ Create a basic file structure for a PolyData"""
        if not self.filename:
            raise RuntimeError("Input filename is not set")

        self.fh = h5.File(self.filename, "w")
        vtkhdf = self.fh.create_group("VTKHDF")
        vtkhdf.attrs.create("Version", self.version)

        typeattr = self.type.encode('ascii')
        typeattr_dtype = h5.string_dtype('ascii', len(typeattr))
        vtkhdf.attrs.create("Type", typeattr,
                            dtype=typeattr_dtype)

        # Groups for geometry
        vtkhdf.create_group("Vertices")
        vtkhdf.create_group("Lines")
        vtkhdf.create_group("Polygons")
        vtkhdf.create_group("Strips")

        # Create groups for data arrays
        vtkhdf.create_group("FieldData")
        vtkhdf.create_group("PointData")
        vtkhdf.create_group("CellData")

        # For time-dependent data
        # self.ntimes defaults to 0, if the TIME_STEPS() is set on the pipeline
        # nsteps is the number of timesteps to write.
        #
        # When using the "manual" WriteNextTime the "nsteps" is 1 in the first
        # step and then incremented as data is being written out
        if self.ntimes > 0:
            self.create_steps()

    def create_steps(self):
        steps = self.fh["VTKHDF"].create_group("Steps")

        # Write the current time index here - let it update as data is
        # being written
        steps.attrs.create("NSteps", 0, dtype=np.int64)

        # More groups for time-dependent offsets
        steps.create_group("FieldData")
        steps.create_group("PointData")
        steps.create_group("CellData")

        steps.create_group("Vertices")
        steps.create_group("Lines")
        steps.create_group("Polygons")
        steps.create_group("Strips")

    def write(self, polydata):
        # REMARK:
        # Storing time in a special "TimeValue" array - not in a FieldData
        if self.ntimes > 0:
            steps = self.fh["VTKHDF/Steps"]
            self.append_offset(steps, "Part", 0, 1)
            self.append_dataset(steps, "TimeValue", np.array((self.time, )))
            steps.attrs["NSteps"] = self.time_index + 1

        # Write geometry if time-dependent
        if self.time_index == 0 or not self.static_geometry:
            self.write_polygeom(polydata)

        # Write arrays
        self.write_data(polydata, "FieldData")
        self.write_data(polydata, "PointData")
        self.write_data(polydata, "CellData")

        if self.time_index > 0:
            # In case geometry or certain arrays are not written out every
            # timestep, we still need to update their offset in the steps-
            # folder - this is a routine that does that
            self.update_offset()

    def write_polygeom(self, polydata):
        vtkhdf = self.fh["VTKHDF"]

        # Write points
        points = polydata.GetPoints()
        buffer = nps.vtk_to_numpy(points.GetData())
        self.append_dataset(vtkhdf, "Points", buffer)

        number = np.array((buffer.shape[0], ), dtype=np.int64)
        self.append_dataset(vtkhdf, "NumberOfPoints", number)

        # Write vertices
        grp = vtkhdf["Vertices"]
        data = polydata.GetVerts()
        buffer = nps.vtk_to_numpy(data.GetOffsetsArray())
        self.append_dataset(grp, "Offsets", buffer)
        buffer = nps.vtk_to_numpy(data.GetConnectivityArray())
        self.append_dataset(grp, "Connectivity", buffer)
        number = np.array((buffer.shape[0], ), dtype=np.int64)
        self.append_dataset(grp, "NumberOfConnectivityIds", number)

        # Write lines
        grp = vtkhdf["Lines"]
        data = polydata.GetLines()
        buffer = nps.vtk_to_numpy(data.GetOffsetsArray())
        self.append_dataset(grp, "Offsets", buffer)
        buffer = nps.vtk_to_numpy(data.GetConnectivityArray())
        self.append_dataset(grp, "Connectivity", buffer)
        number = np.array((buffer.shape[0], ), dtype=np.int64)
        self.append_dataset(grp, "NumberOfConnectivityIds", number)

        # Write polygons
        grp = vtkhdf["Polygons"]
        data = polydata.GetPolys()
        buffer = nps.vtk_to_numpy(data.GetOffsetsArray())
        self.append_dataset(grp, "Offsets", buffer)
        buffer = nps.vtk_to_numpy(data.GetConnectivityArray())
        self.append_dataset(grp, "Connectivity", buffer)
        number = np.array((buffer.shape[0], ), dtype=np.int64)
        self.append_dataset(grp, "NumberOfConnectivityIds", number)

        # Write triangle strips
        grp = vtkhdf["Strips"]
        data = polydata.GetStrips()
        buffer = nps.vtk_to_numpy(data.GetOffsetsArray())
        self.append_dataset(grp, "Offsets", buffer)
        buffer = nps.vtk_to_numpy(data.GetConnectivityArray())
        self.append_dataset(grp, "Connectivity", buffer)
        number = np.array((buffer.shape[0], ), dtype=np.int64)
        self.append_dataset(grp, "NumberOfConnectivityIds", number)

    def write_data(self, polydata, typ):
        if typ == "CellData":
            data = polydata.GetCellData()
        elif typ == "PointData":
            data = polydata.GetPointData()
        elif typ == "FieldData":
            data = polydata.GetFieldData()
        else:
            raise RuntimeError(f"Invalid type: {typ}")

        # Open group CellData/PointData/FieldData
        grp = self.fh["VTKHDF"][typ]

        ndata = data.GetNumberOfArrays()
        for iarr in range(ndata):
            array = data.GetAbstractArray(iarr)
            name = array.GetName()

            buffer = nps.vtk_to_numpy(array)
            self.append_dataset(grp, name, buffer)

    def append_dataset(self, fileh, name, data):
        """Create a dataset if it does not exist, add to end of an existing
        dataset if it exists. Return offset."""

        # Dimensionality checks
        if data.ndim > 2:
            raise RuntimeError

        if name in fileh:
            """ Extend a dataset that is already there """
            dset = fileh[name]

            # More dimensionality checks
            if dset.ndim > 2:
                raise RuntimeError
            if data.ndim != dset.ndim:
                raise RuntimeError
            if data.ndim == 2 and (data.shape[1] != dset.shape[1]):
                raise RuntimeError

            # Offset is present shape of dataset
            offset = dset.shape[0]
            length = data.shape[0]

            # Extend dataset along first dimension
            dset.resize(dset.shape[0] + data.shape[0], axis=0)

            # Write into extended part of dataset
            dset[offset:, ...] = data

        else:
            """ Create a dataset from scratch """
            # Originally the shape is that of the data passed in
            shape = data.shape

            # Maxshape is unlimited in 1st dimension
            maxshape = list(shape)
            maxshape[0] = None

            # Chunksize is more tricky. This is not exact sciecne. We now set
            # it to the size of the data array or 100 kB, whichever is largest.
            # This prevent too small chunks of being written
            minsize = 102400/data.dtype.itemsize
            if data.ndim == 1:
                chunks = (np.maximum(minsize, data.shape[0]), )
            else:
                chunks = (np.maximum(minsize//data.shape[1],
                                     data.shape[0]), data.shape[1])

            # Create and write data
            dset = fileh.create_dataset(name, shape=shape, chunks=chunks,
                                        maxshape=maxshape, data=data)

            offset = 0
            length = data.shape[0]

        # Write entry in "Steps" group
        if self.ntimes > 0:
            self.append_offset(fileh, name, offset, length)

    def append_offset(self, grp, dsetname, offset, length):
        """ Write the offset and length into the 'Steps' datasets. Each time
        a new step is written, the datasets are extended by 1. The chunksize
        is 32 kB, fixed. This will allow for 2048 timesteps in each chunk
        before another chunk is created and being written to. """

        # REMARK:
        # This routine deviate severely from the proposal by Julien:
        # - All array names are postfixed with "Offsets"
        # - Write both offset and length
        # - Same group structure as in the VTKHDF group

        stepsgrp = self.fh["VTKHDF/Steps"]
        for name in ("PointData", "CellData", "FieldData", "Vertices",
                     "Lines", "Polygons", "Strips"):
            if grp.name.endswith(name):
                stepsgrp = stepsgrp[name]
                break
        stepsname = dsetname + "Offsets"

        if stepsname in stepsgrp:
            # Append to already existing array - increase length by one
            data = np.array((offset, length), dtype=np.int64)
            dset = stepsgrp[stepsname]
            dset.resize(dset.shape[0] + 1, axis=0)
            dset[-1, ...] = data
        else:
            # Create Offsets-array for the first time
            data = np.array(((offset, length, ), ), dtype=np.int64)
            maxshape = (None, 2)
            chunks = (2048, 2)
            dset = stepsgrp.create_dataset(stepsname, chunks=chunks,
                                           maxshape=maxshape,
                                           data=data)

    def update_offset(self):
        """ Iterate over all arrays in the Steps-group and update those that
        are not updated at the current timestep. """

        stepsgrp = self.fh["VTKHDF"]["Steps"]
        stepsgrp.visititems(self.update_offset_dset)

    def update_offset_dset(self, name, object):
        # Groups does not have the shape attribute, datasets has
        if hasattr(object, "shape"):
            if object.shape[0] < self.time_index + 1:
                object.resize(object.shape[0] + 1, axis=0)
                object[-1, ...] = object[-2, ...]
        return None


class PythonPolyDataHDFReader(PythonPolyDataHDFBase):

    def __init__(self):
        VTKPythonAlgorithmBase.__init__(self, nInputPorts=0, nOutputPorts=1,
                                        outputType='vtkPolyData')
        super().__init__()

        self.pt_selection = vtk.vtkDataArraySelection()
        self.pt_selection.AddObserver(vtk.vtkCommand.ModifiedEvent,
                                      self.SelectionModifiedCallback)

        self.cl_selection = vtk.vtkDataArraySelection()
        self.cl_selection.AddObserver(vtk.vtkCommand.ModifiedEvent,
                                      self.SelectionModifiedCallback)

        self.selections_is_init = False

        # This acts as a cahche and is shallow-copied to the output
        # every timestep
        self.polydata = vtk.vtkPolyData()

        # Stores the last offset that was read - if the timestep to be read
        # has the same offset - it is not re-read from disk
        self.points_offset = None
        self.array_offset = {
            "PointData": {},
            "CellData": {},
            "FieldData": {}
        }

    def SelectionModifiedCallback(self, selection, event):
        self.Modified()

    def GetCellDataArraySelection(self):
        return self.cl_selection

    def GetPointDataArraySelection(self):
        return self.pt_selection

    def RequestData(self, request, inInfo, outInfo):
        # File should already be open - done in first call to
        # RequestInformation
        executive = self.GetExecutive()
        info = outInfo.GetInformationObject(0)
        if info.Has(executive.UPDATE_TIME_STEP()):
            time = info.Get(executive.UPDATE_TIME_STEP())
        else:
            time = self.time_values[0]

        # Find closest time index to read
        self.time_index = (np.abs(self.time_values - time)).argmin()

        # Read data for that time
        self.read(self.polydata)

        # Shallow-copy to output PolyData
        # https://www.kitware.com/defining-time-varying-sources-with-paraviews-programmable-source/
        outdata = self.GetOutputData(outInfo, 0)
        outdata.ShallowCopy(self.polydata)

        return 1

    def RequestInformation(self, request, inInfo, outInfo):
        """ Called on UpdateInformation """

        # Ensure file is open - this also check if the filename is set
        self.open_file()

        if not self.selections_is_init:
            self.init_selections()

        if self.time_values is None:
            self.GetTimestepValues()

        executive = self.GetExecutive()
        info = self.GetOutputInformation(0)
        info.Remove(executive.TIME_STEPS())
        info.Remove(executive.TIME_RANGE())

        time_range = ((self.time_values[0], self.time_values[-1]))
        info.Set(executive.TIME_RANGE(), time_range, 2)
        info.Set(executive.TIME_STEPS(), self.time_values,
                 self.time_values.shape[0])

        return 1

    def GetOutput(self):
        return self.GetOutputDataObject(0)

    # From PythonCSVReader example:
    # https://kitware.github.io/paraview-docs/latest/python/paraview.util.vtkAlgorithm.html
    def GetTimestepValues(self):
        if self.time_values is None:
            if self.fh is None:
                raise RuntimeError
            self.time_values = self.fh["VTKHDF/Steps/TimeValue"][...]
            self.ntimes = self.time_values.shape[0]

        return self.time_values

    def init_selections(self):
        for arr in self.fh["VTKHDF/PointData"].keys():
            self.pt_selection.AddArray(arr)

        for arr in self.fh["VTKHDF/CellData"].keys():
            self.cl_selection.AddArray(arr)

        self.selections_is_init = True

    def read(self, polydata):
        # Read geometry if time-dependent
        self.read_polygeom(polydata)

        # Read arrays
        self.read_data(polydata, "FieldData")
        self.read_data(polydata, "PointData")
        self.read_data(polydata, "CellData")

    def read_polygeom(self, polydata):
        vtkhdf = self.fh["VTKHDF"]

        # Read points
        offset, length = self.offset_and_length(vtkhdf, "Points")

        # If points have same offset from time to time - assume geometry
        # is stationary and do not re-read:
        #
        # REMARK:
        # This is defineatly a shortcut. Should ideally check points and all
        # four cell classes (points, lines, polys, strips). I'm not sure if
        # this shortcut has any disadvantages?
        if self.points_offset == offset:
            return

        # Update offset
        self.points_offset = offset

        # Read points
        arr = vtkhdf["Points"][offset:offset+length-1, ...]
        points = vtk.vtkPoints()
        points.SetData(nps.numpy_to_vtk(arr, deep=True))
        polydata.SetPoints(points)

        # Read vertices
        data = self.read_cellarray("Vertices")
        polydata.SetVerts(data)

        # Read lines
        data = self.read_cellarray("Lines")
        polydata.SetLines(data)

        # Read polygons
        data = self.read_cellarray("Vertices")
        polydata.SetPolys(data)

        # Read triangle strips
        data = self.read_cellarray("Strips")
        polydata.SetStrips(data)

    def read_cellarray(self, grpname):
        """Reads a cell array (Vertices, lines, polys, strips)"""
        vtkhdf = self.fh["VTKHDF"]
        data = vtk.vtkCellArray()
        grp = vtkhdf[grpname]

        offset, length = self.offset_and_length(grp, "Offsets")
        offsets = grp["Offsets"][offset:offset+length-1]

        offset, length = self.offset_and_length(grp, "Connectivity")
        connectivity = grp["Connectivity"][offset:offset+length-1]

        data.SetData(nps.numpy_to_vtk(offsets, deep=True),
                     nps.numpy_to_vtk(connectivity, deep=True))

        return data

    def read_data(self, polydata, typ):
        if typ == "CellData":
            data = polydata.GetCellData()
        elif typ == "PointData":
            data = polydata.GetPointData()
        elif typ == "FieldData":
            data = polydata.GetFieldData()
        else:
            raise RuntimeError(f"Invalid type: {typ}")

        # Open group CellData/PointData/FieldData
        grp = self.fh["VTKHDF"][typ]

        for arrname in grp.keys():
            if typ == "PointData" and \
                    not self.pt_selection.ArrayIsEnabled(arrname):
                continue
            if typ == "CellData" and \
                    not self.cl_selection.ArrayIsEnabled(arrname):
                continue

            offset, length = self.offset_and_length(grp, arrname)

            # Chek offset and possibly skip reading the array if it is the same
            if arrname in self.array_offset[typ]:
                # Offset is known - check if we can skip
                if self.array_offset[typ][arrname] == offset:
                    continue
            else:
                # Offset is unknown - add it for the next step
                self.array_offset[typ][arrname] = offset

            array = nps.numpy_to_vtk(grp[arrname][offset:offset+length-1, ...],
                                     deep=1)
            array.SetName(arrname)
            data.AddArray(array)

    def offset_and_length(self, grp, dsetname):
        """Return the offset and length for reading a particular array"""
        if self.ntimes > 1:
            stepsgrp = self.fh["VTKHDF/Steps"]
            for name in ("PointData", "CellData", "FieldData", "Vertices",
                         "Lines", "Polygons", "Strips"):
                if grp.name.endswith(name):
                    stepsgrp = stepsgrp[name]
                    break
            steps = stepsgrp[dsetname + "Offsets"]
            offset, length = steps[self.time_index, :]
        else:
            offset = 0
            length = grp[dsetname].shape[0]

        return offset, length


try:
    # These imports are only available if you run the code from within Paraview
    from paraview.util.vtkAlgorithm import smproxy, smproperty, smdomain, \
        smhint

except ImportError:
    # Not executing from within Paraview/pvpytrhon - so the class
    # definitions below are not needed
    pass

else:
    # Decorate the readers and writers for usage within Paraview
    @smproxy.reader(name="PVPythonPolyDataHDFReader",
                    label="Python-based vtkPolyData HDF Reader",
                    extensions=PythonPolyDataHDFReader.extensions,
                    file_description=PythonPolyDataHDFReader.file_description,
                    support_reload=False)
    class PVPythonPolyDataHDFReader(PythonPolyDataHDFReader):
        @smproperty.stringvector(name="FileName", panel_visibility="never")
        @smdomain.filelist()
        @smhint.filechooser(extensions=PythonPolyDataHDFReader.extensions,
                            file_description=PythonPolyDataHDFReader.
                            file_description)
        def SetFileName(self, filename):
            return super().SetFileName(filename)

        @smproperty.dataarrayselection(name="Point Data Arrays")
        def GetPointDataArraySelection(self):
            return super().GetPointDataArraySelection()

        @smproperty.dataarrayselection(name="Cell Data Arrays")
        def GetCellDataArraySelection(self):
            return super().GetCellDataArraySelection()

        @smproperty.doublevector(name="TimestepValues", information_only="1",
                                 si_class="vtkSITimeStepsProperty")
        def GetTimestepValues(self):
            return super().GetTimestepValues()

    @smproxy.writer(name="PVPythonPolyDataHDFWriter",
                    label="Python-based vtkPolyData HDF Writer",
                    extensions=PythonPolyDataHDFWriter.extensions,
                    file_description=PythonPolyDataHDFWriter.file_description,
                    support_reload=False)
    @smproperty.input(name="Input", port_index=0)
    @smdomain.datatype(dataTypes=["vtkPolyData"])
    class PVPythonPolyDataHDFWriter(PythonPolyDataHDFWriter):
        @smproperty.stringvector(name="FileName", panel_visibility="never")
        @smdomain.filelist()
        def SetFileName(self, filename):
            return super().SetFileName(filename)

        @smproperty.xml(xmlstr="""
            <IntVectorProperty name="Geometry is static"
                command="SetStaticGeometry"
                number_of_elements="1"
                default_values="1">
                    <BooleanDomain name="bool"/>
            </IntVectorProperty>""")
        def SetStaticGeometry(self, static_geometry):
            return super().SetStaticGeometry(static_geometry)
