
import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="Van der Pol Simulator", layout="wide")

st.title("Van der Pol Oscillator Simulator")

# Model architecture
class VDPModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)


# Load model and scalers from repository root
@st.cache_resource
def load_model():
    model = VDPModel()

    model.load_state_dict(
        torch.load("van_der_pol_surrogate.pth", map_location="cpu", weights_only=True)
    )
    model.eval()

    input_scaler = joblib.load("scaler_input.pkl")
    output_scaler = joblib.load("scaler_output.pkl")

    return model, input_scaler, output_scaler


try:
    model, input_scaler, output_scaler = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


# User inputs
st.sidebar.header("Simulation Parameters")

mu = st.sidebar.number_input("Mu (μ)", 0.01, 10.0, 1.0, 0.1)
x0 = st.sidebar.number_input("Initial Displacement (x0)", value=2.0)
v0 = st.sidebar.number_input("Initial Velocity (v0)", value=0.0)
t_end = st.sidebar.number_input("Simulation Time", 1.0, 100.0, 30.0)


# Prediction function
def predict(mu, x0, v0, t_end):

    time = np.linspace(0, t_end, 1000)

    inputs = np.column_stack((
        time,
        np.full(1000, mu),
        np.full(1000, x0),
        np.full(1000, v0)
    ))

    inputs = input_scaler.transform(inputs)

    with torch.no_grad():
        predictions = model(
            torch.tensor(inputs, dtype=torch.float32)
        ).numpy()

    predictions = output_scaler.inverse_transform(predictions)

    return time, predictions[:, 0], predictions[:, 1]


# Run simulation
if st.sidebar.button("Run Simulation"):

    time, x, v = predict(mu, x0, v0, t_end)

    st.subheader("Simulation Results")

    # Phase plot
    fig, ax = plt.subplots()
    ax.plot(x, v)
    ax.scatter(x[0], v[0], label="Initial State")
    ax.set_xlabel("Displacement (x)")
    ax.set_ylabel("Velocity (v)")
    ax.set_title("Phase Portrait")
    ax.grid()
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    # Displacement vs time
    fig, ax = plt.subplots()
    ax.plot(time, x)
    ax.set_xlabel("Time")
    ax.set_ylabel("Displacement (x)")
    ax.set_title("Displacement vs Time")
    ax.grid()
    st.pyplot(fig)
    plt.close(fig)

    # Velocity vs time
    fig, ax = plt.subplots()
    ax.plot(time, v)
    ax.set_xlabel("Time")
    ax.set_ylabel("Velocity (v)")
    ax.set_title("Velocity vs Time")
    ax.grid()
    st.pyplot(fig)
    plt.close(fig)

    # Download results
    data = pd.DataFrame({
        "Time": time,
        "Displacement": x,
        "Velocity": v
    })

    st.download_button(
        "Download CSV",
        data.to_csv(index=False),
        "vdp_results.csv",
        "text/csv"
    )
