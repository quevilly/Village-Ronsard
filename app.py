from pyscript import document, when
from js import FileReader, Blob, URL
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("module://matplotlib.backends.html5_canvas_backend")
import matplotlib.pyplot as plt

# -----------------------------
# Paramètres projection ΔT
# -----------------------------
DELTA_T_C = {
    "RCP2.6": {"2050": 1.0, "2100": 1.2},
    "RCP4.5": {"2050": 1.5, "2100": 2.4},
    "RCP6.0": {"2050": 1.8, "2100": 3.0},
    "RCP8.5": {"2050": 2.4, "2100": 4.3},
}

EPW_COLUMNS = [
    "Year","Month","Day","Hour","Minute","Data Source and Uncertainty",
    "Dry Bulb Temperature (C)","Dew Point Temperature (C)","Relative Humidity (%)",
    "Atmospheric Station Pressure (Pa)",
    "Extraterrestrial Horizontal Radiation (Wh/m2)",
    "Extraterrestrial Direct Normal Radiation (Wh/m2)",
    "Horizontal Infrared Radiation Intensity (Wh/m2)",
    "Global Horizontal Radiation (Wh/m2)",
    "Direct Normal Radiation (Wh/m2)","Diffuse Horizontal Radiation (Wh/m2)",
    "Global Horizontal Illuminance (lux)","Direct Normal Illuminance (lux)",
    "Diffuse Horizontal Illuminance (lux)","Zenith Luminance (Cd/m2)",
    "Wind Direction (degrees)","Wind Speed (m/s)","Total Sky Cover (tenths)",
    "Opaque Sky Cover (tenths)","Visibility (km)","Ceiling Height (m)",
    "Present Weather Observation","Present Weather Codes","Precipitable Water (mm)",
    "Aerosol Optical Depth (thousandths)","Snow Depth (cm)","Days Since Last Snowfall",
    "Albedo","Liquid Precipitation Depth (mm)","Liquid Precipitation Quantity (hr)"
]
DATA_COLS = EPW_COLUMNS[6:]

# état
header_lines: list[str] = []
df_epw: pd.DataFrame | None = None
df_proj: pd.DataFrame | None = None
current_file_name: str = ""

# -----------------------------
# Utilitaires EPW
# -----------------------------
def is_data_line(line: str) -> bool:
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 8:
        return False
    try:
        int(parts[0]); int(parts[1]); int(parts[2]); int(parts[3]); int(parts[4])
        return True
    except ValueError:
        return False

def parse_epw_text(text: str) -> tuple[list[str], pd.DataFrame]:
    header = []
    data_rows = []
    for raw in text.splitlines():
        line = raw.rstrip("\n")
        if is_data_line(line):
            data_rows.append([p.strip() for p in line.split(",")])
        else:
            header.append(line)
    if not data_rows:
        raise ValueError("Aucune donnée horaire détectée.")
    # normaliser 35 col
    norm = []
    for row in data_rows:
        if len(row) < len(EPW_COLUMNS):
            row = row + [""] * (len(EPW_COLUMNS) - len(row))
        elif len(row) > len(EPW_COLUMNS):
            row = row[:len(EPW_COLUMNS)]
        norm.append(row)
    df = pd.DataFrame(norm, columns=EPW_COLUMNS)

    # types
    for c in ["Year","Month","Day","Hour","Minute"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    for c in DATA_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # index temporel pour tri; EPW Hour 1..24 -> 0..23
    hour0 = df["Hour"].astype("Int64").fillna(1) - 1
    minute = df["Minute"].astype("Int64").fillna(0)
    dt = pd.to_datetime(dict(
        year=df["Year"].astype(int, errors="ignore"),
        month=df["Month"].astype(int, errors="ignore"),
        day=df["Day"].astype(int, errors="ignore"),
        hour=hour0.astype(int, errors="ignore"),
        minute=minute.astype(int, errors="ignore")
    ), errors="coerce")
    df.index = dt
    df.sort_index(inplace=True)
    return header, df

def write_epw_text(header: list[str], df_out: pd.DataFrame, comment: str) -> str:
    lines = list(header)
    # inject COMMENTS 2
    injected = False
    new_lines = []
    for line in lines:
        if line.upper().startswith("COMMENTS 2"):
            new_lines.append(f"COMMENTS 2,{comment}")
            injected = True
        else:
            new_lines.append(line)
    if not injected:
        new_lines.append(f"COMMENTS 2,{comment}")
    lines = new_lines

    # assurer champs temporels
    idx = df_out.index
    df = df_out.copy()
    df["Year"] = idx.year
    df["Month"] = idx.month
    df["Day"] = idx.day
    df["Hour"] = idx.hour + 1
    df["Minute"] = idx.minute
    if "Data Source and Uncertainty" in df.columns and df["Data Source and Uncertainty"].isna().all():
        df["Data Source and Uncertainty"] = 0
    df = df[EPW_COLUMNS]

    buf = io.StringIO()
    for l in lines:
        buf.write(l.rstrip("\n") + "\n")
    for _, row in df.iterrows():
        vals = []
        for c in EPW_COLUMNS:
            v = row[c]
            if c in ["Year","Month","Day","Hour","Minute","Data Source and Uncertainty"]:
                vals.append(str(int(v)) if pd.notna(v) else "")
            else:
                if pd.isna(v): vals.append("")
                else:
                    try:
                        fv = float(v)
                        if float(fv).is_integer():
                            vals.append(str(int(fv)))
                        else:
                            vals.append(f"{fv:.3f}")
                    except Exception:
                        vals.append(str(v))
        buf.write(",".join(vals) + "\n")
    return buf.getvalue()

# -----------------------------
# Projection
# -----------------------------
def es_hpa(T_c: np.ndarray) -> np.ndarray:
    return 6.112 * np.exp((17.67 * T_c) / (T_c + 243.5))

def dewpoint_from_e(e_hpa: np.ndarray) -> np.ndarray:
    e = np.maximum(e_hpa, 1e-6)
    ln_ratio = np.log(e / 6.112)
    return (243.5 * ln_ratio) / (17.67 - ln_ratio)

def project_df(df: pd.DataFrame, rcp: str, year: str) -> pd.DataFrame:
    dT = DELTA_T_C[rcp][year]
    out = df.copy()
    T = out["Dry Bulb Temperature (C)"].astype(float).to_numpy()
    RH = out["Relative Humidity (%)"].astype(float).to_numpy()
    T_new = T + dT
    e_old = (RH / 100.0) * es_hpa(T)
    RH_new = 100.0 * (e_old / es_hpa(T_new))
    RH_new = np.clip(RH_new, 0, 100)
    Td_new = dewpoint_from_e(e_old)
    out.loc[:, "Dry Bulb Temperature (C)"] = T_new
    if "Relative Humidity (%)" in out.columns:
        out.loc[:, "Relative Humidity (%)"] = RH_new
    if "Dew Point Temperature (C)" in out.columns:
        out.loc[:, "Dew Point Temperature (C)"] = Td_new
    return out

# -----------------------------
# Plot (axe arbitraire 1..N)
# -----------------------------
fig = None
ax = None

def ensure_figure():
    global fig, ax
    if fig is None:
        fig = plt.figure(figsize=(8, 4.5), dpi=110)
    fig.clf()
    ax = fig.add_subplot(111)
    ax.grid(True, alpha=0.3)
    return fig, ax

def plot_variable(name: str, df: pd.DataFrame, dfp: pd.DataFrame):
    fig, ax = ensure_figure()
    n = len(df)
    x = np.arange(1, n + 1, dtype=float)
    y = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
    yp = pd.to_numeric(dfp[name], errors="coerce").to_numpy(dtype=float)

    def dense_line(xv, yv):
        mask = ~np.isnan(yv)
        xv = xv[mask]; yv = yv[mask]
        if len(xv) < 2:
            ax.plot(xv, yv, linewidth=1.0, marker="o", markersize=2)
            return
        xfine = np.linspace(xv.min(), xv.max(), max(1000, len(xv)*5))
        yfine = np.interp(xfine, xv, yv)
        ax.plot(xfine, yfine, linewidth=1.2)
        ax.plot(xv, yv, linestyle="none", marker="o", markersize=2, alpha=0.8)

    # Observé (couleur auto)
    dense_line(x, y)
    # Projection (rouge)
    color_cycle = ax._get_lines.prop_cycler
    ax.plot([], [])  # avance le cycle si besoin
    ax.set_prop_cycle(None)
    # Trace projection en rouge explicitement
    maskp = ~np.isnan(yp)
    xp = x[maskp]; yp2 = yp[maskp]
    if len(xp) >= 2:
        xfine = np.linspace(xp.min(), xp.max(), max(1000, len(xp)*5))
        yfine = np.interp(xfine, xp, yp2)
        ax.plot(xfine, yfine, linewidth=1.4, color="red")
        ax.plot(xp, yp2, linestyle="none", marker="o", markersize=2, color="red", alpha=0.8)
    else:
        ax.plot(xp, yp2, linewidth=1.2, color="red", marker="o", markersize=2)

    rcp = document.querySelector("#rcp").value
    year = document.querySelector("#year").value
    ax.set_title(f"{name} — {rcp} ({year})", fontweight="bold")
    ax.set_xlabel("Unité de temps arbitraire (échantillon)")
    ax.set_ylabel(name)
    ax.legend(["Observé", "Projection"], loc="best")

    # Monte la figure dans la div
    container = document.querySelector("#plotArea")
    # vide
    while container.firstChild:
        container.removeChild(container.firstChild)
    # affiche
    from matplotlib.backends.backend_html5_canvas import FigureCanvasHTML5
    FigureCanvasHTML5(fig, container)

def set_status(msg: str):
    document.querySelector("#status").innerText = msg

def populate_vars_buttons(df: pd.DataFrame):
    vars_div = document.querySelector("#varsList")
    # clear
    while vars_div.firstChild:
        vars_div.removeChild(vars_div.firstChild)
    for col in DATA_COLS:
        btn = document.createElement("button")
        btn.textContent = col
        async def on_click(ev, name=col):
            global df_proj
            rcp = document.querySelector("#rcp").value
            year = document.querySelector("#year").value
            df_proj = project_df(df, rcp, year)
            plot_variable(name, df, df_proj)
            set_status(f"Tracé: {name} ({rcp} {year}).")
            document.querySelector("#exportBtn").disabled = False
        btn.addEventListener("click", on_click)
        vars_div.appendChild(btn)

# -----------------------------
# I/O UI
# -----------------------------
@when("click", "#loadBtn")
def on_load_click(event):
    global header_lines, df_epw, df_proj, current_file_name
    file_input = document.querySelector("#epwfile")
    if not file_input.files or file_input.files.length == 0:
        set_status("Sélectionnez d’abord un fichier EPW.")
        return
    f = file_input.files.item(0)
    current_file_name = f.name
    reader = FileReader.new()
    def onload(evt):
        try:
            text = evt.target.result
            header, df = parse_epw_text(text)
            header_lines[:] = header
            df_epw = df
            df_proj = None
            document.querySelector("#fileInfo").innerText = f.name
            set_status("EPW chargé (tri chronologique appliqué). Choisissez une variable à tracer.")
            populate_vars_buttons(df_epw)
            # reset plot
            ensure_figure()
            container = document.querySelector("#plotArea")
            while container.firstChild:
                container.removeChild(container.firstChild)
            document.querySelector("#exportBtn").disabled = True
        except Exception as e:
            set_status(f"Erreur chargement: {e}")
    reader.onload = onload
    reader.readAsText(f)

@when("click", "#exportBtn")
def on_export_click(event):
    global df_epw, df_proj, header_lines, current_file_name
    if df_epw is None:
        set_status("Chargez un EPW d’abord.")
        return
    if df_proj is None:
        rcp = document.querySelector("#rcp").value
        year = document.querySelector("#year").value
        df_proj = project_df(df_epw, rcp, year)
    rcp = document.querySelector("#rcp").value
    year = document.querySelector("#year").value
    comment = f"Projection simple ΔT: {rcp} / {year} (DryBulb +ΔT; RH recalculée; axe arbitraire pour tracés)"
    text = write_epw_text(header_lines, df_proj, comment)

    blob = Blob.new([text], { "type": "text/plain;charset=utf-8" })
    url = URL.createObjectURL(blob)
    a = document.querySelector("#downloadLink")
    a.href = url
    # Nom de fichier: base + _{rcp}_{year}.epw
    base = (current_file_name.rsplit(".", 1)[0] if current_file_name else "projected")
    a.download = f"{base}_{rcp}_{year}.epw"
    a.style.display = "inline-block"
    a.click()
    set_status("EPW projeté prêt au téléchargement.")
