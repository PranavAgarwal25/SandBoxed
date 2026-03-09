import bpy
import sys

# Start
if __name__ == "__main__":
    argv = sys.argv

    output_path = argv[
        5
    ]

    bpy.ops.export_scene.obj(filepath=output_path)

    exit(0)
