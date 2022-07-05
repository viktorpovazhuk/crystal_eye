from main import main

f = open("config.txt", "r")

properties = []
pict = {}

for i in f.readlines():
    if i == "...\n":
        properties.append(pict)
        pict = {}
    else:
        key, val = i.split("=")[0], i.split("=")[1]
        key = key.strip()
        val = val.strip()

        if key == "path":
            pict[key] = val

        elif key == "target_region":
            coords = val.split(", ")
            pict[key] = (
            int(coords[0].strip()), int(coords[1].strip()), int(coords[2].strip()),
            int(coords[3].strip()))

        elif key == "beta":
            pict[key] = float(val)
        else:
            pict[key] = int(val)

for i in properties:
    main(i["path"], i["target_region"], i["beta"], i["x_step"], i["y_step"],
         i["patch_size"])
