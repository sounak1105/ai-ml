import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import plotly.graph_objects as go
from ripser import ripser
from persim import plot_diagrams
from sklearn.datasets import make_circles
from scipy.spatial.distance import pdist, squareform
import networkx as nx
import pandas as pd

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(page_title="TDA Explorer", layout="wide")

st.title("🔵 Persistent Homology Interactive Explorer")

st.markdown("""
**What is Persistent Homology?**  
As we slowly grow "balls" around each data point (controlled by the radius slider),
topological features appear and disappear:

| Feature | Symbol | Meaning |
|---------|--------|---------|
| Connected Components | **H0** | Clusters of points |
| Loops / Holes | **H1** | Ring-like structures |
| Voids / Cavities | **H2** | Hollow shells (3D only) |

A feature that *persists* for a long radius range is considered **significant**.
""")

st.divider()

# ──────────────────────────────────────────────
# Sidebar — Dataset selection
# ──────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")

dataset_choice = st.sidebar.selectbox(
    "Choose a Dataset",
    [
        "Circle (H1 demo)",
        "Figure-8 (two loops)",
        "Torus (H1 + H2 demo)",
        "Grid",
        "Noisy Line",
        "Random 2D",
        "Sphere (H2 demo)",
        "Upload CSV / XLSX",
    ]
)

n_points = st.sidebar.slider("Number of points (built-in datasets)", 30, 200, 80, step=10)
noise_level = st.sidebar.slider("Noise level", 0.0, 0.2, 0.05, step=0.01)

# ──────────────────────────────────────────────
# Generate / load dataset  (cached in session_state)
# ──────────────────────────────────────────────
# We use a "data key" that encodes every setting that should trigger
# a fresh dataset.  The slider does NOT appear here, so moving the
# slider never regenerates the data.
data_key = (dataset_choice, n_points, noise_level)

def generate_dataset(choice, n, noise):
    """Pure function — only called when dataset settings change."""
    if choice == "Circle (H1 demo)":
        X, _ = make_circles(n_samples=n, noise=noise)

    elif choice == "Figure-8 (two loops)":
        t = np.linspace(0, 2 * np.pi, n)
        x = np.sin(t)
        y = np.sin(t) * np.cos(t)
        X = np.column_stack([x, y]) + np.random.randn(n, 2) * noise

    elif choice == "Torus (H1 + H2 demo)":
        R, r = 3, 1
        u = np.random.uniform(0, 2 * np.pi, n)
        v = np.random.uniform(0, 2 * np.pi, n)
        x = (R + r * np.cos(v)) * np.cos(u)
        y = (R + r * np.cos(v)) * np.sin(u)
        z = r * np.sin(v)
        X = np.vstack([x, y, z]).T + np.random.randn(n, 3) * noise

    elif choice == "Grid":
        side = int(np.sqrt(n))
        gx, gy = np.meshgrid(np.linspace(0, 1, side), np.linspace(0, 1, side))
        X = np.column_stack([gx.ravel(), gy.ravel()])
        X += np.random.randn(len(X), 2) * noise

    elif choice == "Noisy Line":
        x = np.linspace(0, 1, n)
        y = np.random.randn(n) * noise
        X = np.column_stack([x, y])

    elif choice == "Random 2D":
        X = np.random.rand(n, 2)

    elif choice == "Sphere (H2 demo)":
        phi = np.random.uniform(0, np.pi, n)
        theta = np.random.uniform(0, 2 * np.pi, n)
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        X = np.vstack([x, y, z]).T + np.random.randn(n, 3) * noise

    else:
        return None   # Upload handled separately

    return X

X = None
upload_error = None

if dataset_choice != "Upload CSV / XLSX":
    # Only regenerate when dataset settings change, not on every slider move
    if st.session_state.get("data_key") != data_key:
        st.session_state["data_key"] = data_key
        st.session_state["X"] = generate_dataset(dataset_choice, n_points, noise_level)
    X = st.session_state["X"]

if dataset_choice == "Upload CSV / XLSX":
    st.sidebar.markdown("---")
    uploaded_file = st.sidebar.file_uploader(
        "Upload your file", type=["csv", "xlsx", "xls"]
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.sidebar.write("**Preview (first 5 rows):**")
            st.sidebar.dataframe(df.head())

            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

            if len(numeric_cols) < 2:
                upload_error = "Your file needs at least 2 numeric columns."
            else:
                col_x = st.sidebar.selectbox("X column", numeric_cols, index=0)
                col_y = st.sidebar.selectbox("Y column", numeric_cols, index=1)

                use_z = st.sidebar.checkbox("Use a Z column (3D)?", value=False)
                if use_z and len(numeric_cols) >= 3:
                    col_z = st.sidebar.selectbox("Z column", numeric_cols, index=2)
                    X_upload = df[[col_x, col_y, col_z]].dropna().values
                else:
                    X_upload = df[[col_x, col_y]].dropna().values

                # Normalise to [0,1] range so slider works nicely
                X_upload = (X_upload - X_upload.min(axis=0)) / (
                    X_upload.max(axis=0) - X_upload.min(axis=0) + 1e-9
                )

                # Cache uploaded data — upload doesn't re-trigger on slider moves
                upload_key = (uploaded_file.name, col_x, col_y)
                if st.session_state.get("upload_key") != upload_key:
                    st.session_state["upload_key"] = upload_key
                    st.session_state["X_upload"] = X_upload

                X = st.session_state["X_upload"]
                st.sidebar.success(f"Loaded {len(X)} points  ✅")

        except Exception as e:
            upload_error = f"Could not read file: {e}"
    else:
        st.info("👈 Upload a CSV or XLSX file from the sidebar to get started.")

if upload_error:
    st.error(upload_error)

# ──────────────────────────────────────────────
# Main content — only if we have data
# ──────────────────────────────────────────────
if X is not None and len(X) > 1:

    dist = squareform(pdist(X))
    max_dist = float(np.max(dist))

    col_slider, col_info = st.columns([3, 1])
    with col_slider:
        radius = st.slider(
            "🔵 Radius  (filtration parameter ε)",
            min_value=0.01,
            max_value=round(max_dist * 0.6, 2),
            value=round(max_dist * 0.15, 2),
            step=round(max_dist * 0.01, 3),
        )
    with col_info:
        st.metric("Max pairwise distance", f"{max_dist:.3f}")
        st.metric("Points loaded", len(X))

    # ── Colour edges by distance ──────────────────
    def edge_colour(d, radius):
        """Map distance → colour: green (close) → red (far within threshold)."""
        ratio = d / (2 * radius)
        r = int(255 * ratio)
        g = int(255 * (1 - ratio))
        return f"rgb({r},{g},0)"

    # ──────────────────────────────────────────────
    # 2D visualisation
    # ──────────────────────────────────────────────
    if X.shape[1] == 2:

        st.subheader("📐 2D Filtration")

        fig, ax = plt.subplots(figsize=(7, 7))

        # draw circles around points
        for p in X:
            circle = plt.Circle(p, radius, fill=False, alpha=0.15, color="steelblue")
            ax.add_patch(circle)

        # draw edges colour-coded by distance
        n = len(X)
        G = nx.Graph()
        for i in range(n):
            G.add_node(i)

        for i in range(n):
            for j in range(i + 1, n):
                d = dist[i, j]
                if d <= 2 * radius:
                    ratio = d / (2 * radius)
                    color = cm.RdYlGn(1 - ratio)  # green=close, red=far
                    ax.plot(
                        [X[i, 0], X[j, 0]],
                        [X[i, 1], X[j, 1]],
                        color=color,
                        alpha=0.7,
                        linewidth=1,
                    )
                    G.add_edge(i, j)

        ax.scatter(X[:, 0], X[:, 1], color="navy", s=30, zorder=5)

        # colour-bar legend
        sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Edge length  (green=short, red=long)", fontsize=8)

        ax.set_aspect("equal")
        ax.set_title(f"Rips complex at ε = {radius:.3f}")
        st.pyplot(fig)

        cycles = list(nx.cycle_basis(G))
        components = nx.number_connected_components(G)

        c1, c2 = st.columns(2)
        c1.metric("Connected components (H0)", components)
        c2.metric("Detected cycles / loops (H1 hint)", len(cycles))

    # ──────────────────────────────────────────────
    # 3D visualisation
    # ──────────────────────────────────────────────
    else:
        st.subheader("🌐 3D Filtration")

        edges_x, edges_y, edges_z, edge_colors = [], [], [], []
        n = len(X)

        for i in range(n):
            for j in range(i + 1, n):
                d = dist[i, j]
                if d <= 2 * radius:
                    edges_x.extend([X[i, 0], X[j, 0], None])
                    edges_y.extend([X[i, 1], X[j, 1], None])
                    edges_z.extend([X[i, 2], X[j, 2], None])

        points = go.Scatter3d(
            x=X[:, 0], y=X[:, 1], z=X[:, 2],
            mode="markers",
            marker=dict(size=4, color="navy"),
            name="Points",
        )
        edges_trace = go.Scatter3d(
            x=edges_x, y=edges_y, z=edges_z,
            mode="lines",
            line=dict(color="gray", width=1),
            name="Edges",
        )

        fig3d = go.Figure(data=[points, edges_trace])
        fig3d.update_layout(
            height=600,
            title=f"Rips complex at ε = {radius:.3f}",
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        )
        st.plotly_chart(fig3d, use_container_width=True)

    st.divider()

    # ──────────────────────────────────────────────
    # Persistent homology
    # ──────────────────────────────────────────────
    st.subheader("📊 Persistence Diagram")

    st.markdown("""
    Each dot represents a topological feature:
    - **Birth** (x-axis) = radius at which the feature appeared
    - **Death** (y-axis) = radius at which it disappeared
    - Dots far from the diagonal = **significant, long-lasting features**
    """)

    maxdim = 2 if X.shape[1] == 3 else 1
    diagrams = ripser(X, maxdim=maxdim)["dgms"]

    fig2, ax2 = plt.subplots(figsize=(6, 5))
    plot_diagrams(diagrams, show=False, ax=ax2)
    ax2.set_title("Persistence Diagram")
    st.pyplot(fig2)

    # ──────────────────────────────────────────────
    # Raw feature counts
    # ──────────────────────────────────────────────
    st.divider()
    st.subheader("🧮 Total Topological Features (all radii)")

    cols = st.columns(3)
    H0_total = len(diagrams[0])
    H1_total = len(diagrams[1])
    H2_total = len(diagrams[2]) if len(diagrams) > 2 else 0

    cols[0].metric("H0 — Components born", H0_total)
    cols[1].metric("H1 — Loops born", H1_total)
    cols[2].metric("H2 — Voids born", H2_total)

    # ──────────────────────────────────────────────
    # Persistence table
    # ──────────────────────────────────────────────
    with st.expander("📋 View persistence table (birth / death / lifetime)"):
        rows = []
        dim_labels = {0: "H0 (Component)", 1: "H1 (Loop)", 2: "H2 (Void)"}
        for dim, dgm in enumerate(diagrams):
            for pt in dgm:
                birth, death = pt
                lifetime = (death - birth) if not np.isinf(death) else np.inf
                rows.append({
                    "Feature": dim_labels.get(dim, f"H{dim}"),
                    "Birth (ε)": round(float(birth), 4),
                    "Death (ε)": "∞" if np.isinf(death) else round(float(death), 4),
                    "Lifetime": "∞" if np.isinf(lifetime) else round(float(lifetime), 4),
                })
        df_table = pd.DataFrame(rows).sort_values(
            by=["Feature", "Lifetime"], ascending=[True, False]
        )
        st.dataframe(df_table, use_container_width=True)