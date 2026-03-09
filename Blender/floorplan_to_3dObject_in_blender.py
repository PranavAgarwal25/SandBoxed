import bpy
import numpy as np
import json
import sys
import math
import os.path

def read_from_file(file_path):
    with open(file_path + ".txt", "r") as f:
        data = json.loads(f.read())
    return data


def init_object(name):
    mymesh = bpy.data.meshes.new(name)
    myobject = bpy.data.objects.new(name, mymesh)
    bpy.context.collection.objects.link(myobject)
    return myobject, mymesh


def average(lst):
    return sum(lst) / len(lst)


def get_mesh_center(verts):
    x = []
    y = []
    z = []

    for vert in verts:
        x.append(vert[0])
        y.append(vert[1])
        z.append(vert[2])

    return [average(x), average(y), average(z)]


def subtract_center_verts(verts1, verts2):
    for i in range(0, len(verts2)):
        verts2[i][0] -= verts1[0]
        verts2[i][1] -= verts1[1]
        verts2[i][2] -= verts1[2]
    return verts2


def create_custom_mesh(objname, verts, faces, mat=None, cen=None):
    myobject, mymesh = init_object(objname)

    center = get_mesh_center(verts)
    proper_verts = subtract_center_verts(center, verts)

    mymesh.from_pydata(proper_verts, [], faces)
    mymesh.update(calc_edges=True)

    parent_center = [0, 0, 0]
    if cen is not None:
        parent_center = [int(cen[0] / 2), int(cen[1] / 2), int(cen[2])]

    myobject.location.x = center[0] - parent_center[0]
    myobject.location.y = center[1] - parent_center[1]
    myobject.location.z = center[2] - parent_center[2]

    if mat is None:
        myobject.data.materials.append(
            create_mat(np.random.randint(0, 40, size=4))
        )
    else:
        myobject.data.materials.append(mat)
    return myobject


def create_mat(rgb_color):
    mat = bpy.data.materials.new(name="MaterialName")
    mat.diffuse_color = rgb_color
    return mat


def main(argv):

    objs = bpy.data.objects
    objs.remove(objs["Cube"], do_unlink=True)

    if len(argv) > 7:
        program_path = argv[5]
        target = argv[6]
    else:
        exit(0)

    for i in range(7, len(argv)):
        base_path = argv[i]
        create_floorplan(base_path, program_path, i)

    bpy.ops.wm.save_as_mainfile(filepath=program_path + target)


    exit(0)


def create_floorplan(base_path, program_path, name=None):

    if name is None:
        name = 0

    parent, _ = init_object("Floorplan" + str(name))

    path_to_transform_file = program_path + "/" + base_path + "transform"

    transform = read_from_file(path_to_transform_file)

    rot = transform["rotation"]
    pos = transform["position"]
    scale = transform["scale"]

    cen = transform["shape"]

    path_to_data = transform["origin_path"]

    bpy.context.scene.cursor.location = (0, 0, 0)

    path_to_wall_vertical_faces_file = (
        program_path + "/" + path_to_data + "wall_vertical_faces"
    )
    path_to_wall_vertical_verts_file = (
        program_path + "/" + path_to_data + "wall_vertical_verts"
    )

    path_to_wall_horizontal_faces_file = (
        program_path + "/" + path_to_data + "wall_horizontal_faces"
    )
    path_to_wall_horizontal_verts_file = (
        program_path + "/" + path_to_data + "wall_horizontal_verts"
    )

    path_to_floor_faces_file = program_path + "/" + path_to_data + "floor_faces"
    path_to_floor_verts_file = program_path + "/" + path_to_data + "floor_verts"

    path_to_rooms_faces_file = program_path + "/" + path_to_data + "room_faces"
    path_to_rooms_verts_file = program_path + "/" + path_to_data + "room_verts"

    path_to_doors_vertical_faces_file = (
        program_path + "\\" + path_to_data + "door_vertical_faces"
    )
    path_to_doors_vertical_verts_file = (
        program_path + "\\" + path_to_data + "door_vertical_verts"
    )

    path_to_doors_horizontal_faces_file = (
        program_path + "\\" + path_to_data + "door_horizontal_faces"
    )
    path_to_doors_horizontal_verts_file = (
        program_path + "\\" + path_to_data + "door_horizontal_verts"
    )

    path_to_windows_vertical_faces_file = (
        program_path + "\\" + path_to_data + "window_vertical_faces"
    )
    path_to_windows_vertical_verts_file = (
        program_path + "\\" + path_to_data + "window_vertical_verts"
    )

    path_to_windows_horizontal_faces_file = (
        program_path + "\\" + path_to_data + "window_horizontal_faces"
    )
    path_to_windows_horizontal_verts_file = (
        program_path + "\\" + path_to_data + "window_horizontal_verts"
    )

    if (
        os.path.isfile(path_to_wall_vertical_verts_file + ".txt")
        and os.path.isfile(path_to_wall_vertical_faces_file + ".txt")
        and os.path.isfile(path_to_wall_horizontal_verts_file + ".txt")
        and os.path.isfile(path_to_wall_horizontal_faces_file + ".txt")
    ):
        verts = read_from_file(path_to_wall_vertical_verts_file)
        faces = read_from_file(path_to_wall_vertical_faces_file)

        boxcount = 0
        wallcount = 0

        wall_parent, _ = init_object("Walls")

        for walls in verts:
            boxname = "Box" + str(boxcount)
            for wall in walls:
                wallname = "Wall" + str(wallcount)

                obj = create_custom_mesh(
                    boxname + wallname,
                    wall,
                    faces,
                    cen=cen,
                    mat=create_mat((0.5, 0.5, 0.5, 1)),
                )
                obj.parent = wall_parent

                wallcount += 1
            boxcount += 1

        verts = read_from_file(path_to_wall_horizontal_verts_file)
        faces = read_from_file(path_to_wall_horizontal_faces_file)

        boxcount = 0
        wallcount = 0

        for i in range(0, len(verts)):
            roomname = "VertWalls" + str(i)
            obj = create_custom_mesh(
                roomname,
                verts[i],
                faces[i],
                cen=cen,
                mat=create_mat((0.5, 0.5, 0.5, 1)),
            )
            obj.parent = wall_parent

        wall_parent.parent = parent

    if (
        os.path.isfile(path_to_windows_vertical_verts_file + ".txt")
        and os.path.isfile(path_to_windows_vertical_faces_file + ".txt")
        and os.path.isfile(path_to_windows_horizontal_verts_file + ".txt")
        and os.path.isfile(path_to_windows_horizontal_faces_file + ".txt")
    ):
        verts = read_from_file(path_to_windows_vertical_verts_file)
        faces = read_from_file(path_to_windows_vertical_faces_file)

        boxcount = 0
        wallcount = 0

        wall_parent, _ = init_object("Windows")

        for walls in verts:
            boxname = "Box" + str(boxcount)
            for wall in walls:
                wallname = "Wall" + str(wallcount)

                obj = create_custom_mesh(
                    boxname + wallname,
                    wall,
                    faces,
                    cen=cen,
                    mat=create_mat((0.5, 0.5, 0.5, 1)),
                )
                obj.parent = wall_parent

                wallcount += 1
            boxcount += 1

        verts = read_from_file(path_to_windows_horizontal_verts_file)
        faces = read_from_file(path_to_windows_horizontal_faces_file)

        boxcount = 0
        wallcount = 0

        for i in range(0, len(verts)):
            roomname = "VertWindow" + str(i)
            obj = create_custom_mesh(
                roomname,
                verts[i],
                faces[i],
                cen=cen,
                mat=create_mat((0.5, 0.5, 0.5, 1)),
            )
            obj.parent = wall_parent

        wall_parent.parent = parent

    if (
        os.path.isfile(path_to_doors_vertical_verts_file + ".txt")
        and os.path.isfile(path_to_doors_vertical_faces_file + ".txt")
        and os.path.isfile(path_to_doors_horizontal_verts_file + ".txt")
        and os.path.isfile(path_to_doors_horizontal_faces_file + ".txt")
    ):

        verts = read_from_file(path_to_doors_vertical_verts_file)
        faces = read_from_file(path_to_doors_vertical_faces_file)

        boxcount = 0
        wallcount = 0

        wall_parent, _ = init_object("Doors")

        for walls in verts:
            boxname = "Box" + str(boxcount)
            for wall in walls:
                wallname = "Wall" + str(wallcount)

                obj = create_custom_mesh(
                    boxname + wallname,
                    wall,
                    faces,
                    cen=cen,
                    mat=create_mat((0.5, 0.5, 0.5, 1)),
                )
                obj.parent = wall_parent

                wallcount += 1
            boxcount += 1

        verts = read_from_file(path_to_doors_horizontal_verts_file)
        faces = read_from_file(path_to_doors_horizontal_faces_file)

        boxcount = 0
        wallcount = 0

        for i in range(0, len(verts)):
            roomname = "VertWindow" + str(i)
            obj = create_custom_mesh(
                roomname,
                verts[i],
                faces[i],
                cen=cen,
                mat=create_mat((0.5, 0.5, 0.5, 1)),
            )
            obj.parent = wall_parent

        wall_parent.parent = parent

    if os.path.isfile(path_to_floor_verts_file + ".txt") and os.path.isfile(
        path_to_floor_faces_file + ".txt"
    ):

        verts = read_from_file(path_to_floor_verts_file)
        faces = read_from_file(path_to_floor_faces_file)

        cornername = "Floor"
        obj = create_custom_mesh(
            cornername, verts, [faces], mat=create_mat((40, 1, 1, 1)), cen=cen
        )
        obj.parent = parent

        verts = read_from_file(path_to_rooms_verts_file)
        faces = read_from_file(path_to_rooms_faces_file)

        room_parent, _ = init_object("Rooms")

        for i in range(0, len(verts)):
            roomname = "Room" + str(i)
            obj = create_custom_mesh(roomname, verts[i], faces[i], cen=cen)
            obj.parent = room_parent

        room_parent.parent = parent

    if rot is not None:
        parent.rotation_euler = [
            math.radians(rot[0]) + math.pi,
            math.radians(rot[1]),
            math.radians(rot[2]),
        ]

    if pos is not None:
        parent.location.x += pos[0]
        parent.location.y += pos[1]
        parent.location.z += pos[2]

    if scale is not None:
        parent.scale.x = scale[0]
        parent.scale.y = scale[1]
        parent.scale.z = scale[2]


if __name__ == "__main__":
    main(sys.argv)
