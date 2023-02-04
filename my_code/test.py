import os
import vizdoom as vzd

with open(os.path.join(vzd.scenarios_path, "deadly_corridor.cfg"), 'r') as f:

    print(f.readlines()[0])