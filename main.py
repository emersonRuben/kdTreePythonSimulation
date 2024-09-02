import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class Punto:
    def __init__(self, coords):
        self.coords = coords

class NodoKD:
    def __init__(self, punto, izquierda=None, derecha=None):
        self.punto = punto
        self.izquierda = izquierda
        self.derecha = derecha

class KDTree:
    def __init__(self):
        self.raiz = None

    def insertar(self, punto):
        self.raiz = self._insertar(self.raiz, punto, 0)

    def _insertar(self, nodo, punto, profundidad):
        if nodo is None:
            return NodoKD(punto)

        cd = profundidad % len(punto.coords)
        if punto.coords[cd] < nodo.punto.coords[cd]:
            nodo.izquierda = self._insertar(nodo.izquierda, punto, profundidad + 1)
        else:
            nodo.derecha = self._insertar(nodo.derecha, punto, profundidad + 1)

        return nodo

    def eliminar(self, punto):
        self.raiz, _ = self._eliminar(self.raiz, punto, 0)

    def _eliminar(self, nodo, punto, profundidad):
        if nodo is None:
            return nodo, None

        cd = profundidad % len(punto.coords)
        if nodo.punto.coords == punto.coords:
            if nodo.derecha is not None:
                min_punto = self._encontrar_minimo(nodo.derecha, cd, profundidad + 1)
                nodo.punto = min_punto.punto
                nodo.derecha, _ = self._eliminar(nodo.derecha, min_punto.punto, profundidad + 1)
            elif nodo.izquierda is not None:
                return nodo.izquierda, nodo.punto
            else:
                return None, nodo.punto
        elif punto.coords[cd] < nodo.punto.coords[cd]:
            nodo.izquierda, _ = self._eliminar(nodo.izquierda, punto, profundidad + 1)
        else:
            nodo.derecha, _ = self._eliminar(nodo.derecha, punto, profundidad + 1)

        return nodo, None

    def _encontrar_minimo(self, nodo, cd, profundidad):
        if nodo is None:
            return None

        cd_actual = profundidad % len(nodo.punto.coords)
        if cd_actual == cd:
            if nodo.izquierda is None:
                return nodo
            return self._encontrar_minimo(nodo.izquierda, cd, profundidad + 1)
        return min(
            (nodo, 
             self._encontrar_minimo(nodo.izquierda, cd, profundidad + 1),
             self._encontrar_minimo(nodo.derecha, cd, profundidad + 1)),
            key=lambda x: x.punto.coords[cd] if x else float('inf')
        )

    def vecino_mas_cercano(self, objetivo):
        return self._vecino_mas_cercano(self.raiz, objetivo, 0, None)

    def _vecino_mas_cercano(self, nodo, objetivo, profundidad, mejor):
        if nodo is None:
            return mejor

        if mejor is None or self._distancia(nodo.punto, objetivo) < self._distancia(mejor.punto, objetivo):
            mejor = nodo

        cd = profundidad % len(objetivo.coords)
        rama_siguiente = nodo.izquierda if objetivo.coords[cd] < nodo.punto.coords[cd] else nodo.derecha
        rama_otro = nodo.derecha if objetivo.coords[cd] < nodo.punto.coords[cd] else nodo.izquierda

        mejor = self._vecino_mas_cercano(rama_siguiente, objetivo, profundidad + 1, mejor)
        if rama_otro and abs(objetivo.coords[cd] - nodo.punto.coords[cd]) < self._distancia(mejor.punto, objetivo):
            mejor = self._vecino_mas_cercano(rama_otro, objetivo, profundidad + 1, mejor)

        return mejor

    def _distancia(self, p1, p2):
        return sum((p1.coords[i] - p2.coords[i]) ** 2 for i in range(len(p1.coords))) ** 0.5

class InterfazKDTree:
    def __init__(self, root):
        self.tree = KDTree()
        self.root = root
        self.root.title("Búsqueda de Vecinos Cercanos con KD-Tree")
        self.root.state('zoomed')  # Pantalla completa

        # Configuración de la interfaz
        self.frame_controls = tk.Frame(root, width=200)
        self.frame_controls.pack(side=tk.LEFT, fill=tk.Y)

        self.label = tk.Label(self.frame_controls, text="Ingrese las coordenadas del punto (x,y):")
        self.label.pack()

        self.entry = tk.Entry(self.frame_controls)
        self.entry.pack()

        self.boton_insertar = tk.Button(self.frame_controls, text="Insertar Punto", command=self.insertar_punto)
        self.boton_insertar.pack()

        self.boton_eliminar = tk.Button(self.frame_controls, text="Eliminar Punto", command=self.eliminar_punto)
        self.boton_eliminar.pack()

        self.label_buscar = tk.Label(self.frame_controls, text="Ingrese las coordenadas objetivo (x,y):")
        self.label_buscar.pack()

        self.entry_buscar = tk.Entry(self.frame_controls)
        self.entry_buscar.pack()

        self.boton_buscar = tk.Button(self.frame_controls, text="Encontrar Vecino Más Cercano", command=self.buscar_vecino)
        self.boton_buscar.pack()

        self.boton_visualizar = tk.Button(self.frame_controls, text="Visualizar KD-Tree", command=self.visualizar_arbol)
        self.boton_visualizar.pack()

        # Configuración del área de visualización
        self.figura = plt.Figure(figsize=(10, 7), dpi=100)
        self.ax = self.figura.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figura, master=root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.ax.set_title("Visualización del KD-Tree")
        self.ax.set_xlabel("Eje X")
        self.ax.set_ylabel("Eje Y")

    def insertar_punto(self):
        coords = self.entry.get().split(',')
        if len(coords) != 2:
            messagebox.showerror("Error", "Ingrese exactamente dos coordenadas.")
            return

        try:
            coords = [float(coord) for coord in coords]
        except ValueError:
            messagebox.showerror("Error", "Las coordenadas deben ser numéricas.")
            return

        punto = Punto(coords)
        self.tree.insertar(punto)
        self.entry.delete(0, tk.END)
        self.visualizar_arbol()
        messagebox.showinfo("Info", "Punto insertado exitosamente.")

    def eliminar_punto(self):
        coords = self.entry.get().split(',')
        if len(coords) != 2:
            messagebox.showerror("Error", "Ingrese exactamente dos coordenadas.")
            return

        try:
            coords = [float(coord) for coord in coords]
        except ValueError:
            messagebox.showerror("Error", "Las coordenadas deben ser numéricas.")
            return

        punto = Punto(coords)
        self.tree.eliminar(punto)
        self.entry.delete(0, tk.END)
        self.visualizar_arbol()
        messagebox.showinfo("Info", "Punto eliminado exitosamente.")

    def buscar_vecino(self):
        coords = self.entry_buscar.get().split(',')
        if len(coords) != 2:
            messagebox.showerror("Error", "Ingrese exactamente dos coordenadas.")
            return

        try:
            coords = [float(coord) for coord in coords]
        except ValueError:
            messagebox.showerror("Error", "Las coordenadas deben ser numéricas.")
            return

        objetivo = Punto(coords)
        vecino = self.tree.vecino_mas_cercano(objetivo)
        if vecino:
            resultado = f"Vecino Más Cercano: {vecino.punto.coords}"
        else:
            resultado = "No hay puntos en el árbol."
        messagebox.showinfo("Resultado", resultado)

    def visualizar_arbol(self):
        self.ax.clear()
        self.ax.set_title("Visualización del KD-Tree")
        self.ax.set_xlabel("Eje X")
        self.ax.set_ylabel("Eje Y")
        self.ax.grid(True)

        puntos = self._recoger_puntos(self.tree.raiz, [])
        if puntos:
            xs = [punto.coords[0] for punto in puntos]
            ys = [punto.coords[1] for punto in puntos]
            self.ax.scatter(xs, ys, color='blue', label='Puntos')

            self._dibujar_kd_tree(self.tree.raiz, 0, self.ax.get_xlim(), self.ax.get_ylim())
            self.ax.legend()

        self.canvas.draw()

    def _recoger_puntos(self, nodo, puntos):
        if nodo:
            puntos.append(nodo.punto)
            self._recoger_puntos(nodo.izquierda, puntos)
            self._recoger_puntos(nodo.derecha, puntos)
        return puntos

    def _dibujar_kd_tree(self, nodo, profundidad, xlim, ylim):
        if nodo is None:
            return

        cd = profundidad % 2
        if cd == 0:  # División vertical
            self.ax.axvline(x=nodo.punto.coords[0], color='r', linestyle='--')
            if nodo.izquierda:
                self._dibujar_kd_tree(nodo.izquierda, profundidad + 1, (xlim[0], nodo.punto.coords[0]), ylim)
            if nodo.derecha:
                self._dibujar_kd_tree(nodo.derecha, profundidad + 1, (nodo.punto.coords[0], xlim[1]), ylim)
        else:  # División horizontal
            self.ax.axhline(y=nodo.punto.coords[1], color='g', linestyle='--')
            if nodo.izquierda:
                self._dibujar_kd_tree(nodo.izquierda, profundidad + 1, xlim, (ylim[0], nodo.punto.coords[1]))
            if nodo.derecha:
                self._dibujar_kd_tree(nodo.derecha, profundidad + 1, xlim, (nodo.punto.coords[1], ylim[1]))

if __name__ == "__main__":
    root = tk.Tk()
    app = InterfazKDTree(root)
    root.mainloop()
