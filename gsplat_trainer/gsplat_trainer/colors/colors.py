C0 = 0.28209479177387814

def sh_to_rgb(sh):
    return sh * C0 + 0.5

def rgb_to_sh(rgb):
    return (rgb - 0.5) / C0