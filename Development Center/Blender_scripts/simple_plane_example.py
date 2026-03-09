import bpy


def create_custom_mesh(objname, px, py, pz):

    myvertex = []
    myfaces = []

    mypoint = [(-1.0, -1.0, 0.0)]
    myvertex.extend(mypoint)

    mypoint = [(1.0, -1.0, 0.0)]
    myvertex.extend(mypoint)

    mypoint = [(-1.0, 1.0, 0.0)]
    myvertex.extend(mypoint)

    mypoint = [(1.0, 1.0, 0.0)]
    myvertex.extend(mypoint)

    myface = [(0, 1, 3, 2)]
    myfaces.extend(myface)

    mymesh = bpy.data.meshes.new(objname)

    myobject = bpy.data.objects.new(objname, mymesh)

    bpy.context.scene.objects.link(myobject)

    mymesh.from_pydata(myvertex, [], myfaces)
    mymesh.update(calc_edges=True)

    myobject.location.x = px
    myobject.location.y = py
    myobject.location.z = pz

    return myobject


curloc = bpy.context.scene.cursor_location

create_custom_mesh("Awesome_object", curloc[0], curloc[1], 0)
