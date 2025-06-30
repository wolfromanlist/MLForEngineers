import numpy as np
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display, clear_output

class RegressionWidget():
    def __init__(self, x, y):
        self.x = x
        self.y_real = y
        self.init_w = 1.0
        self.init_b = 0.0
        self.w_slider = widgets.FloatSlider(value=self.init_w, min=1, max=4, step=0.1, description="w")
        self.b_slider = widgets.FloatSlider(value=self.init_b, min=-1, max=4, step=0.1, description="b")
        self.output = widgets.Output()

    @staticmethod
    def my_line(w, b, x):
        return w * x + b

    @staticmethod
    def mse(y_real, y_predicted):
        return 1 / len(y_real) * np.sum((y_real - y_predicted) ** 2)


    def update(self, change=None):
        with self.output:
            clear_output(wait=True)
            w = self.w_slider.value
            b = self.b_slider.value
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=self.x, y=self.y_real, mode='markers', name='Daten'))
            fig.add_trace(go.Scatter(x=self.x, y=self.my_line(w, b, self.x), mode='lines', name='Linie', line=dict(color='blue')))
            fig.update_layout(
                title={"text": f"MSE: {self.mse(self.y_real, self.my_line(w, b, self.x)):.2f}", "x": 0.5},
                xaxis_title="x",
                yaxis_title="y",
                width=800,
                height=500
            )
            fig.show()

    def show(self):
        # Callbacks verbinden
        self.w_slider.observe(self.update, names="value")
        self.b_slider.observe(self.update, names="value")

        # Anzeigen
        display(widgets.HBox([self.b_slider, self.w_slider]))
        display(self.output)

        # Initiales Zeichnen
        self.update()
