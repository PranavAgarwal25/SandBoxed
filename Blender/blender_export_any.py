import bpy
import os
import sys

if __name__ == "__main__":
    argv = sys.argv

    input_path = argv[5]
    bpy.ops.wm.open_mainfile(filepath=input_path)

    format = argv[6]
    output_path = argv[
        7
    ]

    if format == ".obj":
        bpy.ops.export_scene.obj(filepath=output_path)
    elif format == ".fbx":
        bpy.ops.export_scene.fbx(filepath=output_path)
    elif format == ".gltf":
        bpy.ops.export_scene.gltf(filepath=output_path, export_format='GLTF_SEPARATE')
    elif format == ".x3d":
        bpy.ops.export_scene.x3d(filepath=output_path)
    elif format == ".blend":
        bpy.ops.wm.save_as_mainfile(filepath=output_path)
    else:
        bpy.ops.export_scene.obj(filepath=output_path)

    exit(0)
