import numpy as np
import plotly.graph_objects as go
import ipywidgets as widgets
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider, RadioButtons, Checkbox, VBox, HBox


def plot_two_point_loss(y1, y2, yhat1, yhat2, show_bce=True, show_mse=False, show_likelihood=True):
    # Einzelverluste
    bce1 = - (y1 * np.log(yhat1) + (1 - y1) * np.log(1 - yhat1))
    bce2 = - (y2 * np.log(yhat2) + (1 - y2) * np.log(1 - yhat2))
    mse1 = (yhat1 - y1)**2
    mse2 = (yhat2 - y2)**2
    lik1 = yhat1 if y1 == 1 else 1 - yhat1
    lik2 = yhat2 if y2 == 1 else 1 - yhat2

    # Gesamtloss / Likelihood
    total_bce = (bce1 + bce2) / 2
    total_mse = (mse1 + mse2) / 2
    total_likelihood = lik1 * lik2

    text = f"""
    Einzelwerte  
    y₁ = {y1}, ŷ₁ = {yhat1:.2f} → L₁ = {lik1:.3f}, BCE₁ = {bce1:.3f}  
    y₂ = {y2}, ŷ₂ = {yhat2:.2f} → L₂ = {lik2:.3f}, BCE₂ = {bce2:.3f}

    Gesamtwerte  
    ➤ Likelihood: L = L₁ × L₂ = {total_likelihood:.5f}  
    ➤ BCE: {(bce1 + bce2)/2:.3f}  
    ➤ MSE: {(mse1 + mse2)/2:.3f}
    """

    fig = go.Figure()

    if show_bce:
        fig.add_trace(go.Bar(name="BCE", x=["y₁", "y₂"], y=[bce1, bce2], marker_color="blue"))
        fig.add_trace(go.Bar(name="BCE Gesamt", x=["Gesamter Loss"], y=[total_bce], marker_color="lightblue"))

    if show_mse:
        fig.add_trace(go.Bar(name="MSE", x=["y₁", "y₂"], y=[mse1, mse2], marker_color="green"))
        fig.add_trace(go.Bar(name="MSE Gesamt", x=["Gesamter Loss"], y=[total_mse], marker_color="lightgreen"))

    if show_likelihood:
        fig.add_trace(go.Bar(name="Likelihood", x=["y₁", "y₂"], y=[lik1, lik2], marker_color="orange"))
        fig.add_trace(go.Bar(name="Likelihood Gesamt", x=["Gesamte Likelihood"], y=[total_likelihood], marker_color="gold"))

    fig.update_layout(
        title=f"Loss und Likelihood für zwei Datenpunkte",
        barmode='group',
        height=500, width=800,
        yaxis_title="Wert",
    )

    fig.show()
    print(text)

def LikelihoodWidget():
    interact(
    plot_two_point_loss,
    y1=RadioButtons(options=[0, 1], value=1, description='y₁:'),
    yhat1=FloatSlider(value=0.8, min=0.001, max=0.999, step=0.001, description="ŷ₁"),
    y2=RadioButtons(options=[0, 1], value=0, description='y₂:'),
    yhat2=FloatSlider(value=0.2, min=0.001, max=0.999, step=0.001, description="ŷ₂"),
    show_bce=Checkbox(value=True, description="BCE"),
    show_mse=Checkbox(value=False, description="MSE"),
    show_likelihood=Checkbox(value=True, description="Likelihood"),
);


# Datenbeispiel
x = np.array([0.2, 0.3, 0.5, 0.8])
y = np.array([0, 0, 1, 1])

# Sigmoidfunktion
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Binary Cross-Entropy
def bce(y_true, y_pred):
    epsilon = 1e-15  # zur Vermeidung von log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Plot-Funktion
def update_plot(w, b):
    z = w * x + b
    y_hat = sigmoid(z)
    loss = bce(y, y_hat)

    fig = go.Figure()

    # Datenpunkte mit Farben nach Label
    fig.add_trace(go.Scatter(
        x=x, y=y, mode='markers',
        marker=dict(color=y, colorscale='Bluered', size=10),
        name='True Labels',
        showlegend=False  # Legende für die Punkte ausblenden
    ))

    # Zwei Dummy-Traces für die Legende: Blau für y=0, Rot für y=1
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(color='blue', size=10),
        name='True Label: 0'
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(color='red', size=10),
        name='True Label: 1'
    ))

    # Vorhersagekurve
    x_range = np.linspace(0, 1, 200)
    z_range = w * x_range + b
    y_pred_range = sigmoid(z_range)

    fig.add_trace(go.Scatter(x=x_range, y=y_pred_range, mode='lines',
                            line=dict(color='black'), name='Vorhersagekurve'))

    # BCE-Text als Annotation
    fig.update_layout(
        title=f"Vorhersagekurve & Datenpunkte<br>BCE = {loss:.4f}",
        xaxis_title='x',
        yaxis_title='y / ŷ',
        yaxis=dict(range=[-0.1, 1.1]),
        legend=dict(x=0.02, y=0.98)
    )
    
    return fig

def BCE_Widget():
    # Interaktive Sliders
    w_slider = widgets.FloatSlider(value=5, min=-15, max=15, step=0.1, description='w')
    b_slider = widgets.FloatSlider(value=-2, min=-10, max=10, step=0.1, description='b')

    # Interaktives Plot-Update
    out = widgets.Output()
    def interactive_update(change=None):
        with out:
            out.clear_output(wait=True)
            fig = update_plot(w_slider.value, b_slider.value)
            fig.show()

    # Events registrieren
    w_slider.observe(interactive_update, names='value')
    b_slider.observe(interactive_update, names='value')

    # Initiales Anzeigen
    interactive_update()

    VBox([HBox([w_slider, b_slider]), out])

    return VBox([HBox([w_slider, b_slider]), out])

def sigmoid_widget():
    def plot_sigmoid(w=1.0, b=0.0):
        x = np.linspace(-10, 10, 100)
        y = 1 / (1 + np.exp(-w * (x - b)))
        plt.figure(figsize=(8, 5))
        plt.plot(x, y, label=f'Sigmoid (w={w:.2f}, b={b:.2f})', color='blue')
        plt.axhline(0.5, color='red', linestyle='--', linewidth=1, label='y=0.5')
        plt.axvline(b, color='gray', linestyle='--', linewidth=1, label=f'x={b:.2f}')
        plt.xlim(-10, 10)
        plt.ylim(-0.1, 1.1)
        plt.legend()
        plt.title("Sigmoid Funktion (interaktiv)")
        plt.xlabel("x")
        plt.ylabel("Sigmoid(x)")
        plt.grid(True)
        plt.show()

    interact(plot_sigmoid,
            w=FloatSlider(value=0.0, min=-3.0, max=3.0, step=0.1, description='Parameter w'),
            b=FloatSlider(value=0.0, min=-5.0, max=5.0, step=0.1, description='Parameter b'))
    
def likelihood_widget():
    def plot_likelihood(p=0.5):
        x = np.linspace(0, 1, 100)
        y = lambda x: x**13 * (1-x)**7
        fig, ax = plt.subplots(figsize=(5,5))
        ax.plot(x, y(x), label=f'L = {y(p):.3g}')
        ax.axvline(p, color='red', linestyle='--', label=f'p = {p:.2f}')
        ax.set_xlabel('p')
        ax.set_ylabel('Likelihood')
        ax.legend()
        plt.show()

    interact(plot_likelihood, 
            p=FloatSlider(value=0.5, min=0, max=1, step=0.01, description='p'))