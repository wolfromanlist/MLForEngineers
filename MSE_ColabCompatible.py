import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output

# Zufallsdaten
generator = np.random.default_rng(2948292983384)
x = np.linspace(0, 20, 20)
y_real = 2 * x + 1 + 10 * (generator.random(20) - 0.5)

def my_line(w, b):
    return w * x + b

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

class InteractivePlot:
    def __init__(self, n=5):
        self.n = n
        self.w_slider = widgets.FloatSlider(
            value=1.0,
            min=1.0,
            max=3.0,
            step=2/(n-1),
            description='w',
            continuous_update=False
        )
        self.b_slider = widgets.FloatSlider(
            value=0.0,
            min=0.0,
            max=2.0,
            step=2/(n-1),
            description='b',
            continuous_update=False
        )
        self.reset_button = widgets.Button(description='Reset')

    def plot(self, w, b):
        y_pred = my_line(w, b)
        error = mse(y_real, y_pred)
        clear_output(wait=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        # Regressionslinie und Datenpunkte
        ax1.plot(x, y_pred, label=f"Line: y = {w:.2f}x + {b:.2f}", color='blue')
        ax1.scatter(x, y_real, label='Data', color='black')
        ax1.set_title(f'MSE: {error:.2f}')
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.legend()
        ax1.grid(True)

        # w-b Raum
        ax2.set_xlim([1.0, 3.0])
        ax2.set_ylim([0.0, 2.0])
        ax2.set_xticks(np.linspace(1.0, 3.0, self.n))
        ax2.set_yticks(np.linspace(0.0, 2.0, self.n))
        ax2.grid(True, which='both', linestyle='--', alpha=0.5)
        ax2.scatter(w, b, marker='x', color='red')
        ax2.set_xlabel('w')
        ax2.set_ylabel('b')
        ax2.set_title('Parameterraum')

        plt.tight_layout()
        plt.show()

    def reset(self, _):
        self.w_slider.value = 1.0
        self.b_slider.value = 0.0

    def __call__(self):
        out = widgets.interactive_output(
            self.plot, {'w': self.w_slider, 'b': self.b_slider}
        )
        self.reset_button.on_click(self.reset)
        controls = widgets.HBox([self.w_slider, self.b_slider, self.reset_button])
        display(widgets.VBox([controls, out]))
