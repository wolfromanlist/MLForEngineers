import numpy as np
import plotly.graph_objects as go
from ipywidgets import interact, FloatSlider, RadioButtons, Checkbox


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

def disp():
    interact(
    plot_two_point_loss,
    y1=RadioButtons(options=[0, 1], value=1, description='y₁:'),
    yhat1=FloatSlider(value=0.8, min=0.01, max=0.99, step=0.01, description="ŷ₁"),
    y2=RadioButtons(options=[0, 1], value=0, description='y₂:'),
    yhat2=FloatSlider(value=0.2, min=0.01, max=0.99, step=0.01, description="ŷ₂"),
    show_bce=Checkbox(value=True, description="BCE"),
    show_mse=Checkbox(value=False, description="MSE"),
    show_likelihood=Checkbox(value=True, description="Likelihood"),
);

