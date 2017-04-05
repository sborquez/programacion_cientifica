def twotothree(line, cell):
    ip = get_ipython()
    original = cell.replace("raw_input","input").split("\n")
    python3 = ""
    for line in original:
        if "print" in line:
            line = line.replace("print ", "print(") + ")"
        python3 += line + "\n"
    #print("Version Python3:")
    #print(python3)
    ip.run_code(python3)

def load_ipython_extension(ipython):
    ipython.register_magic_function(twotothree, 'cell')
