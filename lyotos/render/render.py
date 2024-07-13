import vtkmodules.vtkInteractionStyle
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkCommonColor import vtkNamedColors

from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import (
    vtkCellArray,
    vtkPolyData,
    vtkPolyLine
)

from vtkmodules.vtkCommonTransforms import (
    vtkMatrixToHomogeneousTransform,
    vtkTransform,
)
from vtkmodules.vtkFiltersGeneral import vtkTransformFilter, vtkTransformPolyDataFilter
from vtkmodules.vtkFiltersSources import vtkConeSource, vtkDiskSource

from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkDataSetMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer
)

colors = vtkNamedColors()


def build_system(system):
    #
    # Now we create an instance of vtkConeSource and set some of its
    # properties. The instance of vtkConeSource "cone" is part of a
    # visualization pipeline (it is a source process object) it produces data
    # (output type is vtkPolyData) which other filters may process.
    #
    cone = vtkConeSource()
    cone.SetHeight(3.0)
    cone.SetRadius(1.0)
    cone.SetResolution(10)

    retval = []

    for surf in system.surfaces:
        source = vtkDiskSource()

        source.SetOuterRadius(100)
        source.SetInnerRadius(0)
        source.SetCircumferentialResolution(100)

        transform = vtkMatrixToHomogeneousTransform()
        
        transform.SetInput(surf._cs.toGCS.vtk)

        f = vtkTransformPolyDataFilter()
        f.SetInputConnection(source.GetOutputPort())
        f.SetTransform(transform)

        
        mapper = vtkDataSetMapper()
        actor = vtkActor()

        mapper.SetInputConnection(f.GetOutputPort())
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(colors.GetColor3d('Blue'))
        
        retval += [ actor  ]
        print(surf)

    return retval
    

def render(trace):

    actors = build_system(trace._system)
    
    
    ren1 = vtkRenderer()

    for actor in actors:   
        ren1.AddActor(actor)

    for path in trace._paths:

        ren1.AddActor(actor)
        
    ren1.SetBackground(colors.GetColor3d('MidnightBlue'))
    
    # Finally we create the render window which will show up on the screen.
    # We put our renderer into the render window using AddRenderer. We also
    # set the size to be 300 pixels by 300.
    #
    
    
