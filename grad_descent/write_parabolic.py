import numpy as np

if __name__ == '__main__':
    with open("parabola.csv", "wt") as f:
        f.write("x,y\n")
        for x in np.linspace(-3, 3, 20):
            f.write(f"{x:3.8},{x**2:4.8}\n")