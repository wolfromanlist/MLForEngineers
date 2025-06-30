""" import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display

class GradientDescentVisualizer:
    def __init__(self, steps=5, w_range=(1.0, 3.0), b_range=(0.0, 2.0), eta_range=(0.0001, 0.005)):
        # Data
        self.x = np.linspace(0, 20, 20)
        rng = np.random.default_rng(2948292983384)
        self.y = 2 * self.x + 1 + 10 * (rng.random(20) - 0.5)

        # Parameters
        self.steps = steps
        self.init_w, self.init_b, self.init_eta = 1.0, 1.0, 0.0015
        self.w_range, self.b_range, self.eta_range = w_range, b_range, eta_range

        # Precompute meshgrid and MSE surface
        w_vals = np.linspace(*w_range, 100)
        b_vals = np.linspace(*b_range, 100)
        self.W, self.B = np.meshgrid(w_vals, b_vals, indexing='ij')
        self.Z = self.mse(self.y, self.y_hat(self.W, self.B))

        self._setup_widgets()
        self._setup_figure()
        self._link_callbacks()

    def y_hat(self, w, b):
        x = self.x
        if np.isscalar(w) and np.isscalar(b):
            return w * x + b
        return x[:, None, None] * w[None, :, :] + b[None, :, :]

    def mse(self, y_true, y_pred):
        if y_pred.ndim == 1:
            return np.mean((y_true - y_pred) ** 2)
        return np.mean((y_true[:, None, None] - y_pred) ** 2, axis=0)

    def compute_descent_path(self, w0, b0, eta):
        w, b = w0, b0
        history = [(w, b)]
        errors = [self.mse(self.y, self.y_hat(w, b))]

        for _ in range(self.steps):
            y_pred = self.y_hat(w, b)
            dw = -2 * np.mean(self.x * (self.y - y_pred))
            db = -2 * np.mean(self.y - y_pred)
            w -= eta * dw
            b -= eta * db
            history.append((w, b))
            errors.append(self.mse(self.y, self.y_hat(w, b)))
        return history, errors

    def _setup_widgets(self):
        self.w_slider = widgets.FloatSlider(value=self.init_w, min=self.w_range[0], max=self.w_range[1], step=0.1, description='w₀:')
        self.b_slider = widgets.FloatSlider(value=self.init_b, min=self.b_range[0], max=self.b_range[1], step=0.1, description='b₀:')
        self.eta_slider = widgets.FloatSlider(value=self.init_eta, min=self.eta_range[0], max=self.eta_range[1], step=0.0001, readout_format='.4f',description='η:')
        self.reset_button = widgets.Button(description="Reset")

    def _setup_figure(self):
        self.fig = go.FigureWidget(make_subplots(rows=1, cols=2, subplot_titles=("Loss over Time", "Parameter Space")))
        self.fig.update_layout(
            xaxis=dict(range=[-0.2, self.steps + .2], fixedrange=True),
            yaxis=dict(range=[0, 175], fixedrange=True),
            xaxis2=dict(range=[self.w_range[0], self.w_range[1]], fixedrange=True),
            yaxis2=dict(range=[self.b_range[0], self.b_range[1]], fixedrange=True),
        )

        self.history, self.errors = self.compute_descent_path(self.init_w, self.init_b, self.init_eta)
        ws, bs = zip(*self.history)

        contour = go.Contour(
            x=np.linspace(*self.w_range, 100),
            y=np.linspace(*self.b_range, 100),
            z=self.Z.T,
            colorscale='Viridis',
            contours=dict(showlabels=True),
            colorbar=dict(title='Loss')
        )

        self.loss_trace = go.Scatter(
            x=list(range(len(self.errors))),
            y=self.errors,
            mode='lines+markers',
            name='Loss'
        )

        arrow_trace = go.Scatter(
            x=ws,
            y=bs,
            mode='lines+markers',
            marker=dict(
                symbol=['circle'] + ['triangle-up'] * (len(ws) - 1),
                size=[0] + [12] * (len(ws) - 1),
                color='red',
                angle=[0] * len(ws),
                angleref='previous'
            ),
            line=dict(color='red'),
            name='Descent Path',
            showlegend=True
        )

        self.fig.add_trace(self.loss_trace, row=1, col=1)
        self.fig.update_xaxes(title_text="Step", row=1, col=1)
        self.fig.update_yaxes(title_text="Loss", row=1, col=1)

        self.fig.add_trace(contour, row=1, col=2)
        self.fig.update_xaxes(title_text="w", row=1, col=2)
        self.fig.update_yaxes(title_text="b", row=1, col=2)

        self.arrow_trace_index = len(self.fig.data)
        self.fig.add_trace(arrow_trace, row=1, col=2)

        self.fig.update_layout(height=600, width=1000, showlegend=False)


    def _update_plot(self, w0, b0, eta):
        self.history, self.errors = self.compute_descent_path(w0, b0, eta)
        ws, bs = zip(*self.history)

        with self.fig.batch_update():
            print('update figure')
            self.fig.data[0].x = list(range(len(self.errors)))
            self.fig.data[0].y = self.errors

            trace = self.fig.data[self.arrow_trace_index]
            trace.x = ws
            trace.y = bs
            trace.marker.angle = [0] * len(ws)
            trace.marker.angleref = 'previous'


    def _reset(self, *_):
        self.w_slider.value = self.init_w
        self.b_slider.value = self.init_b
        self.eta_slider.value = self.init_eta

    def _link_callbacks(self):
        self.reset_button.on_click(self._reset)
    

    def show(self):
        controls = widgets.VBox([
            self.w_slider,
            self.b_slider,
            self.eta_slider,
            self.reset_button
        ])

        out = widgets.interactive_output(
            self._update_plot,
            {
                'w0': self.w_slider,
                'b0': self.b_slider,
                'eta': self.eta_slider
            }
        )

        display(controls)
        display(self.fig)
        display(out)

"""

""" 
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display

class GradientDescentVisualizer:
    def __init__(self, steps=5, w_range=(1.0, 3.0), b_range=(0.0, 2.0), eta_range=(0.0001, 0.005)):
        # Daten
        self.x = np.linspace(0, 20, 20)
        rng = np.random.default_rng(2948292983384)
        self.y = 2 * self.x + 1 + 10 * (rng.random(20) - 0.5)

        # Parameter
        self.steps = steps
        self.init_w, self.init_b, self.init_eta = 1.0, 1.0, 0.0015
        self.w_range, self.b_range, self.eta_range = w_range, b_range, eta_range

        # Vorberechnung Meshgrid & MSE-Oberfläche
        w_vals = np.linspace(*self.w_range, 100)
        b_vals = np.linspace(*self.b_range, 100)
        W_mesh, B_mesh = np.meshgrid(w_vals, b_vals, indexing='ij')
        Z_mse = self.mse(self.y, self.y_hat(W_mesh, B_mesh))

        # Static contour trace erstellen (wird nie neu berechnet)
        self.contour_trace_spec = go.Contour(
            x=w_vals,
            y=b_vals,
            z=Z_mse.T,
            colorscale='Viridis',
            contours=dict(showlabels=True),
            colorbar=dict(title='Loss')
        )

        # Widgets
        self._setup_widgets()
        self.plot_output_area = widgets.Output()
        self._link_callbacks()

    def y_hat(self, w, b):
        x = self.x
        if np.isscalar(w) and np.isscalar(b):
            return w * x + b
        return x[:, None, None] * w[None, :, :] + b[None, :, :]

    def mse(self, y_true, y_pred):
        if y_pred.ndim == 1:
            return np.mean((y_true - y_pred) ** 2)
        return np.mean((y_true[:, None, None] - y_pred) ** 2, axis=0)

    def compute_descent_path(self, w0, b0, eta):
        w, b = w0, b0
        history, errors = [(w, b)], [self.mse(self.y, self.y_hat(w, b))]
        for _ in range(self.steps):
            y_pred = self.y_hat(w, b)
            dw = -2 * np.mean(self.x * (self.y - y_pred))
            db = -2 * np.mean(self.y - y_pred)
            w -= eta * dw; b -= eta * db
            history.append((w, b))
            errors.append(self.mse(self.y, self.y_hat(w, b)))
        return history, errors

    def _setup_widgets(self):
        self.w_slider = widgets.FloatSlider(value=self.init_w,
                                           min=self.w_range[0], max=self.w_range[1],
                                           step=0.1, description='w₀:')
        self.b_slider = widgets.FloatSlider(value=self.init_b,
                                           min=self.b_range[0], max=self.b_range[1],
                                           step=0.1, description='b₀:')
        self.eta_slider = widgets.FloatSlider(value=self.init_eta,
                                             min=self.eta_range[0], max=self.eta_range[1],
                                             step=0.0001, readout_format='.4f',
                                             description='η:')
        self.reset_button = widgets.Button(description="Reset")

    def _create_figure(self, ws, bs, errors):
        fig = go.Figure(make_subplots(rows=1, cols=2,
                                      subplot_titles=("Loss over Time", "Parameter Space")))
        # Static contour (aus Specifications)
        fig.add_trace(self.contour_trace_spec, row=1, col=2)

        # Loss-Trace
        fig.add_trace(go.Scatter(x=list(range(len(errors))), y=errors,
                                 mode='lines+markers', name='Loss'),
                      row=1, col=1)
        fig.update_xaxes(title_text="Step", row=1, col=1)
        fig.update_yaxes(title_text="Loss", row=1, col=1)

        # Descent Path
        fig.add_trace(go.Scatter(x=ws, y=bs,
                                 mode='lines+markers',
                                 marker=dict(symbol=['circle'] + ['triangle-up']*(len(ws)-1),
                                             size=[6] + [12]*(len(ws)-1), color='red',
                                             angle=[0]*len(ws), angleref='previous'),
                                 line=dict(color='red'), name='Path'),
                      row=1, col=2)
        fig.update_xaxes(title_text="w", row=1, col=2)
        fig.update_yaxes(title_text="b", row=1, col=2)

        fig.update_layout(height=600, width=1000,
                          showlegend=False, title_text='Gradient Descent')
        return fig

    def _update_plot(self, w0, b0, eta):
        history, errors = self.compute_descent_path(w0, b0, eta)
        ws, bs = zip(*history)
        new_fig = self._create_figure(ws, bs, errors)
        with self.plot_output_area:
            self.plot_output_area.clear_output(wait=True)
            display(new_fig)

    def _reset(self, *_):
        self.w_slider.value = self.init_w
        self.b_slider.value = self.init_b
        self.eta_slider.value = self.init_eta

    def _link_callbacks(self):
        widgets.interactive_output(self._update_plot,
                                   {'w0': self.w_slider, 'b0': self.b_slider, 'eta': self.eta_slider})
        self.reset_button.on_click(self._reset)

    def show(self):
        controls = widgets.VBox([self.w_slider, self.b_slider, self.eta_slider, self.reset_button])
        display(widgets.VBox([controls, self.plot_output_area]))
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display

class GradientDescentVisualizer:
    def __init__(self, steps=5, w_range=(1.0, 3.0), b_range=(0.0, 2.0), eta_range=(0.0001, 0.005)):
        # Daten und Parameter
        self.x = np.linspace(0, 20, 20)
        rng = np.random.default_rng(2948292983384)
        self.y = 2 * self.x + 1 + 10 * (rng.random(20) - 0.5)
        self.steps = steps
        self.init_w, self.init_b, self.init_eta = 1.0, 1.0, 0.0015
        self.w_range, self.b_range, self.eta_range = w_range, b_range, eta_range

        # Vorberechnung Meshgrid & MSE-Oberfläche für Contour-Caching (50×50 statt 100×100)
        w_vals = np.linspace(*self.w_range, 17)
        b_vals = np.linspace(*self.b_range, 17)
        W_mesh, B_mesh = np.meshgrid(w_vals, b_vals, indexing='ij')
        Z_mse = self.mse(self.y, self.y_hat(W_mesh, B_mesh))

        # Statisches Contour-Trace Spec ohne Labels
        self.contour_spec = go.Contour(
            x=w_vals, y=b_vals, z=Z_mse.T,
            colorscale='Viridis',
            contours=dict(showlabels=False),  # Kontur-Beschriftungen aus
            colorbar=dict(title='Loss')
        )

        # Widgets aufsetzen
        self._setup_widgets()
        self.plot_output = widgets.Output()
        self._link_callbacks()

    def y_hat(self, w, b):
        if np.isscalar(w) and np.isscalar(b):
            return w * self.x + b
        return self.x[:, None, None] * w[None, :, :] + b[None, :, :]

    def mse(self, y_true, y_pred):
        if y_pred.ndim == 1:
            return np.mean((y_true - y_pred) ** 2)
        return np.mean((y_true[:, None, None] - y_pred) ** 2, axis=0)

    def compute_descent_path(self, w0, b0, eta):
        w, b = w0, b0
        history, errors = [(w, b)], [self.mse(self.y, self.y_hat(w, b))]
        for _ in range(self.steps):
            y_pred = self.y_hat(w, b)
            dw = -2 * np.mean(self.x * (self.y - y_pred))
            db = -2 * np.mean(self.y - y_pred)
            w -= eta * dw; b -= eta * db
            history.append((w, b))
            errors.append(self.mse(self.y, self.y_hat(w, b)))
        return history, errors

    def _setup_widgets(self):
        self.w_slider = widgets.FloatSlider(value=self.init_w,
                                           min=self.w_range[0], max=self.w_range[1],
                                           step=0.1, description='w₀:')
        self.b_slider = widgets.FloatSlider(value=self.init_b,
                                           min=self.b_range[0], max=self.b_range[1],
                                           step=0.1, description='b₀:')
        self.eta_slider = widgets.FloatSlider(value=self.init_eta,
                                             min=self.eta_range[0], max=self.eta_range[1],
                                             step=0.0001, readout_format='.4f', description='η:')
        self.reset_button = widgets.Button(description="Reset")

    def _create_figure(self, ws, bs, errors):
        fig = go.Figure(make_subplots(rows=1, cols=2,
                                      subplot_titles=("Loss over Time", "Parameter Space")))
        # Statisches Contour
        fig.add_trace(self.contour_spec, row=1, col=2)

        # Loss-Trace mit WebGL
        fig.add_trace(go.Scattergl(
            x=list(range(len(errors))), y=errors,
            mode='lines+markers', name='Loss'
        ), row=1, col=1)
        fig.update_xaxes(title_text="Step", row=1, col=1)
        fig.update_yaxes(title_text="Loss", row=1, col=1)

        # Descent Path mit Standard-Scatter für Winkel- und Pfeildarstellung
        fig.add_trace(go.Scatter(
            x=ws, y=bs,
            mode='lines+markers',
            marker=dict(
                symbol=['circle'] + ['triangle-up']*(len(ws)-1),
                size=[6] + [12]*(len(ws)-1),
                color='red',
                angle=[0]*len(ws),     # gleiche Winkel für Konsistenz
                angleref='previous'    # unterstützt nur bei go.Scatter
            ),
            line=dict(color='red'), name='Path'
        ), row=1, col=2)
        # Feste Achsenbereiche
        fig.update_xaxes(title_text="w", row=1, col=2,
                         range=[self.w_range[0]-0.1, self.w_range[1]+0.1], fixedrange=True)
        fig.update_yaxes(title_text="b", row=1, col=2,
                         range=[self.b_range[0]-0.1, self.b_range[1]+0.1], fixedrange=True)

        # Layout nur einmal setzen
        fig.layout.update(height=400, width=680, showlegend=False, title='Gradient Descent')
        return fig

    def _update_plot(self, w0, b0, eta):
        history, errors = self.compute_descent_path(w0, b0, eta)
        ws, bs = zip(*history)
        with self.plot_output:
            self.plot_output.clear_output(wait=True)
            fig = self._create_figure(ws, bs, errors)
            display(fig)

    def _reset(self, *_):
        self.w_slider.value = self.init_w
        self.b_slider.value = self.init_b
        self.eta_slider.value = self.init_eta

    def _link_callbacks(self):
        widgets.interactive_output(
            self._update_plot,
            {'w0': self.w_slider, 'b0': self.b_slider, 'eta': self.eta_slider}
        )
        self.reset_button.on_click(self._reset)

    def show(self):
        controls = widgets.VBox([self.w_slider, self.b_slider, self.eta_slider, self.reset_button])
        display(controls)
        # Initiales Rendern
        with self.plot_output:
            hist, errs = self.compute_descent_path(self.init_w, self.init_b, self.init_eta)
            ws0, bs0 = zip(*hist)
            display(self._create_figure(ws0, bs0, errs))
        display(self.plot_output)
