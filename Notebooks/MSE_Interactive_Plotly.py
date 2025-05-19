""" import numpy as np
import plotly.graph_objects as go
import plotly.colors
import ipywidgets as widgets
from IPython.display import display
from plotly.subplots import make_subplots

class LinearRegressionVisualizer:
    def __init__(self, n = 5):
        # Daten erzeugen
        self.generator = np.random.default_rng(2948292983384)
        self.x = np.linspace(0, 20, 20)
        self.y = 2 * self.x + 1 + 10 * (self.generator.random(20) - 0.5)

        # Gitter definieren
        self.n = n
        self.w_vals = np.linspace(1, 3, self.n)
        self.b_vals = np.linspace(0, 2, self.n)
        self.W, self.B = np.meshgrid(self.w_vals, self.b_vals)
        self.Z = np.array([[self._mse(self.y, self._line(w, b, self.x)) for w in self.w_vals] for b in self.b_vals])
        self.min_mse = self.Z.min()
        self.max_mse = self.Z.max()
        self.increment = self.w_vals[1] - self.w_vals[0]
        self.half_rect = self.increment / 2
        self.colorscale = plotly.colors.sequential.Viridis

        self.w_edges = np.linspace(self.w_vals[0] - self.half_rect, self.w_vals[-1] + self.half_rect, self.n + 1)
        self.b_edges = np.linspace(self.b_vals[0] - self.half_rect, self.b_vals[-1] + self.half_rect, self.n + 1)

        # Widgets
        self.w_slider = widgets.FloatSlider(
            value=self.w_vals[0],
            min=self.w_vals[0],
            max=self.w_vals[-1],
            step=self.increment,
            description='w'
        )
        self.b_slider = widgets.FloatSlider(
            value=self.b_vals[0],
            min=self.b_vals[0],
            max=self.b_vals[-1],
            step=self.increment,
            description='b'
        )

        self.reset_button = widgets.Button(description="Reset", button_style='warning')
        self.reset_button.on_click(self._reset)

        # Buchführung
        self.visited_points = set()
        self.bounds_list = []

        # Plot vorbereiten
        self._init_figure()
        self._update_plot()

        # Eventbindung
        self.w_slider.observe(self._update_plot, names="value")
        self.b_slider.observe(self._update_plot, names="value")

    def _line(self, m, b, input_x):
        return m * input_x + b

    def _mse(self, y_real, y_pred):
        return np.mean((y_real - y_pred) ** 2)

    def _normalize_mse(self, val):
        return (val - self.min_mse) / (self.max_mse - self.min_mse)

    def _mse_to_color(self, val):
        norm_val = self._normalize_mse(val)
        index = int(norm_val * (len(self.colorscale) - 1))
        return self.colorscale[index]

    def _init_figure(self):
        self.fig = make_subplots(rows=1, cols=2, subplot_titles=("Regression", "Parameterraum"), horizontal_spacing=0.15)
        self.fig = go.FigureWidget(self.fig)

        # Regressionsplot
        self.line = go.Scatter(x=self.x, y=self._line(self.w_slider.value, self.b_slider.value, self.x),
                               mode='lines', showlegend=False)
        self.scatter = go.Scatter(x=self.x, y=self.y, mode='markers',
                                  showlegend=False, marker=dict(color='blue', size=10))
        self.fig.add_trace(self.line, row=1, col=1)
        self.fig.add_trace(self.scatter, row=1, col=1)

        # Dummy-Colorbar
        self.color_dummy = go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(
                colorscale=self.colorscale,
                cmin=self.min_mse,
                cmax=self.max_mse,
                color=[self.min_mse],
                colorbar=dict(
                    title="MSE",
                    tickmode="linear",
                    tick0=self.min_mse,
                    dtick=(self.max_mse - self.min_mse) / 5,
                ),
                showscale=True
            ),
            showlegend=False
        )
        self.fig.add_trace(self.color_dummy, row=1, col=2)

        self.fig.add_annotation(
            dict(
                x=1.04,
                y=0,  # Will be updated
                xref='paper',
                yref='paper',
                showarrow=False,
                text='▶',
                font=dict(size=18, color='black'),
                name='colorbar_marker'
            )
        )

        self.cross_marker = go.Scatter(
            x=[self.w_slider.value],
            y=[self.b_slider.value],
            mode="markers",
            marker=dict(symbol="x", color="black", size=12),
            showlegend=False,
            zorder=5
        )
        self.fig.add_trace(self.cross_marker, row=1, col=2)

        self.fig.update_layout(
            height=600,
            width=1100,
            xaxis=dict(range=[-2, 22], title="x", fixedrange=True),
            yaxis=dict(range=[-2, 47], title="y", fixedrange=True),
            xaxis2=dict(
                range=[1 - self.half_rect, 3 + self.half_rect],
                title="w",
                fixedrange=True,
                tickvals=np.arange(
                    self.w_vals[0],
                    self.w_vals[-1] + self.increment/2,
                    self.increment
                ),
                showgrid=False,
                gridcolor='lightgray',
                gridwidth=1
            ),
            yaxis2=dict(
                range=[0 - self.half_rect, 2 + self.half_rect],
                title="b",
                fixedrange=True,
                tickvals=np.arange(
                    self.b_vals[0],
                    self.b_vals[-1] + self.increment/2,
                    self.increment
                ),
                showgrid=False,
                gridcolor='lightgray',
                gridwidth=1,
                zeroline=False
            )
        )

        # Liniengitter
        for w in self.w_edges:
            self.fig.add_shape(type="line", x0=w, x1=w,
                               y0=self.b_edges[0], y1=self.b_edges[-1],
                               line=dict(color="lightgray", width=1),
                               xref="x2", yref="y2", layer="below")
        for b in self.b_edges:
            self.fig.add_shape(type="line", x0=self.w_edges[0], x1=self.w_edges[-1],
                               y0=b, y1=b,
                               line=dict(color="lightgray", width=1),
                               xref="x2", yref="y2", layer="below")

    def _update_plot(self, change=None):
        w = self.w_slider.value
        b = self.b_slider.value
        point_key = (w, b)

        y_pred = self._line(w, b, self.x)
        current_mse = self._mse(self.y, y_pred)
        norm_mse = 0.956 * self._normalize_mse(current_mse) - 0.005

        # Update Annotation
        for annotation in self.fig.layout.annotations:
            if getattr(annotation, 'name', '') == 'colorbar_marker':
                annotation.y = norm_mse
                annotation.text = f"{current_mse:.2f} ▶"
                break

        self.fig.data[0].y = y_pred
        self.fig.layout.annotations[0].text = f"MSE: {current_mse:.2f}"

        if point_key not in self.visited_points:
            x0 = w - self.half_rect
            x1 = w + self.half_rect
            y0 = b - self.half_rect
            y1 = b + self.half_rect
            color = self._mse_to_color(current_mse)

            rect = go.Scatter(
                x=[x0, x1, x1, x0, x0],
                y=[y0, y0, y1, y1, y0],
                fill="toself",
                fillcolor=color,
                line=dict(width=0),
                mode="lines",
                showlegend=False
            )
            self.fig.add_trace(rect, row=1, col=2)
            self.visited_points.add(point_key)

            # Tick-Werte updaten
            if current_mse not in self.bounds_list:
                self.bounds_list.append(current_mse)
                self.bounds_list.sort()

                step = 1
                if len(self.bounds_list) > 20:
                    step = 5
                elif len(self.bounds_list) > 5:
                    step = 2

                self.fig.data[2].marker.colorbar.tickvals = self.bounds_list[::step]
                self.fig.data[2].marker.colorbar.ticktext = [f"{v:.2f}" for v in self.bounds_list[::step]]

        self.fig.data[3].x = [w]
        self.fig.data[3].y = [b]

    def _reset(self, _=None):
        # Reset slider values to default
        self.w_slider.value = self.w_vals[0]
        self.b_slider.value = self.b_vals[0]

        # Clear all rectangle patches (those added after the static elements)
        self.fig.data = self.fig.data[:4]  # Keep original: regression line, points, color dummy, cross marker

        # Clear visited state and bounds
        self.visited_points.clear()
        self.bounds_list.clear()

        # Reset colorbar ticks
        self.fig.data[2].marker.colorbar.tickvals = []
        self.fig.data[2].marker.colorbar.ticktext = []

        # Reset the annotation marker
        for annotation in self.fig.layout.annotations:
            if getattr(annotation, 'name', '') == 'colorbar_marker':
                annotation.text = "▶"
                annotation.y = 0
                break

        # Update everything
        self._update_plot()

    def show(self):
        display(widgets.HBox([self.w_slider, self.b_slider, self.reset_button]), self.fig)


    #def show(self):
    #    display(self.w_slider, self.b_slider, self.fig)
"""
import numpy as np
import plotly.graph_objects as go
import plotly.colors
import ipywidgets as widgets
from IPython.display import display
from plotly.subplots import make_subplots

class LinearRegressionVisualizer:
    def __init__(self, n = 5):
        self.plot_out = widgets.Output()

        self.generator = np.random.default_rng(2948292983384)
        self.x = np.linspace(0, 20, 20)
        self.y = 2 * self.x + 1 + 10 * (self.generator.random(20) - 0.5)

        self.n = n
        self.w_vals = np.linspace(1, 3, self.n)
        self.b_vals = np.linspace(0, 2, self.n)
        self.W, self.B = np.meshgrid(self.w_vals, self.b_vals)
        self.Z = np.array([[self._mse(self.y, self._line(w, b, self.x)) for w in self.w_vals] for b in self.b_vals])
        self.min_mse = self.Z.min()
        self.max_mse = self.Z.max()
        self.increment = self.w_vals[1] - self.w_vals[0]
        self.half_rect = self.increment / 2
        self.colorscale = plotly.colors.sequential.Viridis
        self.color_grid = np.array([[self._mse_to_color(self.Z[i, j]) for j in range(self.n)] for i in range(self.n)])

        self.w_edges = np.linspace(self.w_vals[0] - self.half_rect, self.w_vals[-1] + self.half_rect, self.n + 1)
        self.b_edges = np.linspace(self.b_vals[0] - self.half_rect, self.b_vals[-1] + self.half_rect, self.n + 1)

        self.w_slider = widgets.FloatSlider(value=self.w_vals[0], min=self.w_vals[0], max=self.w_vals[-1], step=self.increment, description='w')
        self.b_slider = widgets.FloatSlider(value=self.b_vals[0], min=self.b_vals[0], max=self.b_vals[-1], step=self.increment, description='b')
        self.reset_button = widgets.Button(description="Reset", button_style='warning')
        self.reset_button.on_click(self._reset)

        self.visited_points = set()

        self.base_shapes = []
        for w_val in self.w_edges:
            self.base_shapes.append(dict(type="line", x0=w_val, x1=w_val, y0=self.b_edges[0], y1=self.b_edges[-1],
                                        line=dict(color="lightgray", width=1), xref="x2", yref="y2", layer="below"))
        for b_val in self.b_edges:
            self.base_shapes.append(dict(type="line", x0=self.w_edges[0], x1=self.w_edges[-1], y0=b_val, y1=b_val,
                                        line=dict(color="lightgray", width=1), xref="x2", yref="y2", layer="below"))


        self._update_plot()

        self.w_slider.observe(self._update_plot, names="value")
        self.b_slider.observe(self._update_plot, names="value")

    def _line(self, m, b, input_x):
        return m * input_x + b

    def _mse(self, y_real, y_pred):
        return np.mean((y_real - y_pred) ** 2)

    def _normalize_mse(self, val):
        return (val - self.min_mse) / (self.max_mse - self.min_mse)

    def _mse_to_color(self, val):
        norm_val = self._normalize_mse(val)
        index = int(norm_val * (len(self.colorscale) - 1))
        return self.colorscale[index]

    def _update_plot(self, change=None):
        with self.plot_out:
            self.plot_out.clear_output(wait=True)

            w = self.w_slider.value
            b = self.b_slider.value
            point_key = (w, b)
            idx_w = int(round((w - self.w_vals[0]) / self.increment))
            idx_b = int(round((b - self.b_vals[0]) / self.increment))
            current_mse = self.Z[idx_b, idx_w]

            if point_key not in self.visited_points:
                self.visited_points.add(point_key)

            norm_mse = 0.956 * self._normalize_mse(current_mse) - 0.005

            fig = make_subplots(rows=1, cols=2, subplot_titles=("Regression", "Parameterraum"), horizontal_spacing=0.15)

            fig.add_trace(go.Scatter(x=self.x, y=self._line(w, b, self.x), mode='lines', showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=self.x, y=self.y, mode='markers', showlegend=False, marker=dict(color='blue', size=10)), row=1, col=1)

            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(colorscale=self.colorscale, cmin=self.min_mse, cmax=self.max_mse, color=[self.min_mse], colorbar=dict(title="MSE"), showscale=True), showlegend=False), row=1, col=2)

            fig.add_trace(go.Scatter(x=[w], y=[b], mode="markers", marker=dict(symbol="x", color="black", size=12), showlegend=False), row=1, col=2)

            fig.add_annotation(dict(x=1.04, y=norm_mse, xref='paper', yref='paper', showarrow=False, text=f"{current_mse:.2f} ▶", font=dict(size=18, color='black')))

            fig.update_layout(
                height=600,
                width=850,
                xaxis=dict(range=[-2, 22], title="x", fixedrange=True),
                yaxis=dict(range=[-2, 47], title="y", fixedrange=True),
                xaxis2=dict(
                    range=[1 - self.half_rect, 3 + self.half_rect],
                    title="w",
                    fixedrange=True,
                    tickvals=np.arange(self.w_vals[0], self.w_vals[-1] + self.increment/2, self.increment),
                    showgrid=False,
                    gridcolor='lightgray',
                    gridwidth=1
                ),
                yaxis2=dict(
                    range=[0 - self.half_rect, 2 + self.half_rect],
                    title="b",
                    fixedrange=True,
                    tickvals=np.arange(self.b_vals[0], self.b_vals[-1] + self.increment/2, self.increment),
                    showgrid=False,
                    gridcolor='lightgray',
                    gridwidth=1,
                    zeroline=False
                )
            )

            dynamic_rectangle_shapes = []
            for (w_val, b_val) in self.visited_points:
                idx_w = int(round((w_val - self.w_vals[0]) / self.increment))
                idx_b = int(round((b_val - self.b_vals[0]) / self.increment))
                color = self.color_grid[idx_b, idx_w]

                dynamic_rectangle_shapes.append(dict(
                    type="rect",
                    x0=w_val - self.half_rect, y0=b_val - self.half_rect,
                    x1=w_val + self.half_rect, y1=b_val + self.half_rect,
                    xref="x2", yref="y2",
                    line=dict(width=0),
                    fillcolor=color,
                    layer='below'
                ))

            fig.update_layout(shapes=self.base_shapes + dynamic_rectangle_shapes)

            display(fig)

    def _reset(self, _=None):
        self.w_slider.value = self.w_vals[0]
        self.b_slider.value = self.b_vals[0]
        self.visited_points.clear()
        self._update_plot()

    def show(self):
        display(widgets.HBox([self.w_slider, self.b_slider, self.reset_button]), self.plot_out)
