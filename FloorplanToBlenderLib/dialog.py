from pyfiglet import Figlet




def figlet(text="Floorplan to Blender3d", font="slant"):
    f = Figlet(font=font)
    print(f.renderText(text))


def init():
    print("----- CREATE BLENDER PROJECT FROM FLOORPLAN WITH DIALOG -----")
    print("Welcome to this program. Please answer the questions below to progress.")
    print("Remember that you can change data more efficiently in the config file.")
    print("")


def question(text, default):

    return input(text + " [default = " + default + "]: ")


def end_copyright():
    print("")
    print("FloorplanToBlender3d Copyright (C) 2022  Daniel Westberg")
    print("This program comes with ABSOLUTELY NO WARRANTY;")
    print(

    )
    print("")
