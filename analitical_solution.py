import matplotlib.pyplot as plt
import numpy as np
from pylab import cm
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle


"""
Beam parameters: 
L=6m
E=20 000 MPa
h=800mm
P=150 kN
"""
L=6000
E=20000
h=800
P=-150

def v(x):
    return ((L-x/3.)*6.*P*x**2)/(E*h**3)

def plot_v():
    x = np.linspace(start=0, stop=6000, num=1000)
    y = [v(i) for i in x]
    y = np.array(y)
    fig, ax = plt.subplots()
    ax.set_xlim(0, max(x))  # Ustawienie zakresu osi x od 0 do maksymalnej wartości x

    ax.plot(x,y)
    plt.title('deflection function v(x)')
    plt.xlabel('x [mm]')
    plt.ylabel('v [mm]')
    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ratio = 1/4.
    ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
    plt.show()

def M(x):
    return P*x - P*L

def plot_M():
    x = np.linspace(start=0, stop=6000, num=1000)
    y = [M(i) for i in x]
    y = np.array(y)

    plt.plot(x, y)
    plt.title('bending moment M(x)')
    plt.xlabel('x [mm]')
    plt.ylabel('M(x)')
    plt.show()

def T(x):
    return P
def plot_T():
    x = np.linspace(start=0, stop=6000, num=1000)
    y = [T(i) for i in x]
    y = np.array(y)

    plt.plot(x, y)
    plt.title('sheer force T(x)')
    plt.xlabel('x [mm]')
    plt.ylabel('T [kN]')
    plt.show()

def sigma(x,y):
   return 12.*(L-x)*P*y/h**3

def plot_sigma():
    x = np.arange(0, 6000, 6)
    y = np.arange(-400, 400, 6)
    X, Y = np.meshgrid(x, y)
    s = sigma(X,Y)

    #plt.figure(figsize=(12, 4))
    plt.figure(figsize=(12, 4))
    im=plt.imshow(s, extent=[0,6000,-400,400])

    values = np.unique(s.ravel())
    def GetSpacedElements(array, numElems=4):
        out = array[np.round(np.linspace(0, len(array) - 1, numElems)).astype(int)]
        return out
    values_10 = GetSpacedElements(values, numElems=10)
    colors = [im.cmap(im.norm(value)) for value in values_10]

    patches = [mpatches.Patch(color=colors[i], label=f"{np.round(values_10[i],decimals=2)}") for i in range(len(values_10))]
    patches[:0] = [mpatches.Patch(color='none', label=r'$\sigma$ [kN/mm]')]
    #plt.legend(handles=patches,bbox_to_anchor=(1.12, 1), borderaxespad=0.)
    plt.legend(handles=patches,loc="upper left", bbox_to_anchor=(1.024, 1.03), draggable=True) # loc mowi ktorym kornerem legendy przesuwasz
    plt.title('normal stress '+r'$\sigma$'+'(x, y)')
    plt.xlabel('x [mm]')
    plt.ylabel('y [mm]')
    plt.tight_layout() # powoduje ze okno zawsze obejmuje legende bez wzgl na jej pozycje
    plt.show()


def tau(x,y):
    return P * 6 * (h/2 - y)* (h/2 +y)/h**3

def plot_tau():
    x = np.arange(0, 6000, 6)
    y = np.arange(-400, 400, 6)
    X, Y = np.meshgrid(x, y)
    t = tau(X,Y)
    plt.figure(figsize=(12, 4))
    im = plt.imshow(t, extent=[0,6000,-400,400])

    plt.title('sheer stress '+r'$\tau$'+'(x, y)')
    plt.xlabel('x [mm]')
    plt.ylabel('y [mm]')
    values = np.unique(t.ravel())
    def GetSpacedElements(array, numElems=4):
        out = array[np.round(np.linspace(0, len(array) - 1, numElems)).astype(int)]
        return out
    values_10 = GetSpacedElements(values, numElems=10)
    colors = [im.cmap(im.norm(value)) for value in values_10]
    patches = [mpatches.Patch(color=colors[i], label=f"{np.round(values_10[i], decimals=2)}") for i in range(len(values_10))]
    patches[:0] = [mpatches.Patch(color='none', label=r'$\tau$ [kN/mm]')]
    #plt.legend(handles=patches,loc=2,bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    # plt.legend(handles=patches,loc="upper right")#, bbox_to_anchor=(1.25, 1.0))
    plt.legend(handles=patches,loc="upper left", bbox_to_anchor=(1.024, 1.03), draggable=True) # loc mowi ktorym kornerem legendy przesuwasz
    plt.tight_layout() # powoduje ze okno zawsze obejmuje legende bez wzgl na jej pozycje

    plt.show()

def T(x):
    return P

def plot_T():
    x = np.linspace(start=0, stop=6000, num=1000)
    y = [T(i) for i in x]
    y = np.array(y)
    fig, ax = plt.subplots()
    ax.plot(x,y)
    plt.title('sheer force T(x)')
    plt.xlabel('x [mm]')
    plt.ylabel('T [kN]')
    ax.set_xlim(0, max(x))  # Ustawienie zakresu osi x od 0 do maksymalnej wartości x
    ax.set_ylim(0, max(y)+0.1*max(y))  # Ustawienie zakresu osi y od 0 do maksymalnej wartości y
    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ratio = 1/4.
    ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
    plt.show()

def M(x):
    return (P*x)-P*L

def plot_M():
    x = np.linspace(start=0, stop=6000, num=1000)
    y = [M(i) for i in x]
    y = np.array(y)
    fig, ax = plt.subplots()
    ax.plot(x,y)
    plt.title('bending moment M(x)')
    plt.xlabel('x [mm]')
    plt.ylabel('M [Nm]')
    ax.set_xlim(0, max(x))  # Ustawienie zakresu osi x od 0 do maksymalnej wartości x
    ax.set_ylim(0, max(y)+0.2*max(y))  # Ustawienie zakresu osi y od 0 do maksymalnej wartości y
    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ratio = 1/4.
    ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
    plt.tight_layout() # powoduje ze okno zawsze obejmuje legende bez wzgl na jej pozycje
    plt.show()

plot_tau()
plot_sigma()
plot_v()
plot_T()
plot_M()