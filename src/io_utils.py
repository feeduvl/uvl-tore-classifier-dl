import json
import os


def saveContentToFile(path, file, content):
    if os.path.isfile(path) and os.access(path, os.R_OK):
        # checks if file exists
        print("File exists and is readable")
    else:
        print("Either file is missing or is not readable, creating file...")
        with open(os.path.join(path, file), 'w') as fout:
            json.dump(content, fout)