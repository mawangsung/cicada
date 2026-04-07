"""ODIN QuantumLogicGate v2.0 — Gradio Web Application.

Deploy on HuggingFace Spaces:
  - Set HF_TOKEN and HF_DATASET_REPO as Space Secrets
  - app.py at repo root is the Spaces entry point

Run locally:
  export HF_TOKEN=hf_...
  export HF_DATASET_REPO=username/odin-sensor-data
  python app.py
"""

import os
import tempfile
from pathlib import Path

import gradio as gr

# ── ODIN core ────────────────────────────────────────────────────────────────
from odin.mappings.futhark import ELDER_FUTHARK, RUNE_ORDER
from odin.mappings.hexagrams import HEXAGRAMS, encode_hexagram
from odin.gates.entanglement import HuginnMuninnMetin
from odin.state.register import QuantumRegister
from odin.state.qubit import Qubit
from odin.visualization.bloch import BlochSphere
from odin.sensor.lidar import LiDARInput, SUPPORTED_FORMATS as LIDAR_FORMATS
from odin.sensor.dashcam import DashcamInput, SUPPORTED_FORMATS as VIDEO_FORMATS

# ── HF hub ───────────────────────────────────────────────────────────────────
from odin.hub import HFDatasetBridge, RuneOracle

# ── Runtime config ────────────────────────────────────────────────────────────
HF_TOKEN        = os.environ.get("HF_TOKEN", "")
HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO", "")
HF_CIRCUIT_REPO = os.environ.get("HF_CIRCUIT_REPO", "")

engine = HuginnMuninnMetin()
bridge = HFDatasetBridge(token=HF_TOKEN or None)
oracle = RuneOracle(token=HF_TOKEN or None)

# ── Rune checkbox choices (char + name + meaning) ────────────────────────────
TWO_QUBIT_RUNES = {"ᚷ", "ᛇ", "ᛉ", "ᛖ"}
RUNE_CHOICES = [
    (f"{ELDER_FUTHARK[c].char}  {ELDER_FUTHARK[c].name} — {ELDER_FUTHARK[c].meaning}", c)
    for c in RUNE_ORDER
]


# ── Tab 1: Rune Circuit ───────────────────────────────────────────────────────

def run_rune_circuit(
    rune_chars: list[str],
    initial_state: str,
    push_to_hf: bool,
):
    # Build initial register
    if initial_state == "GHZ  (|000⟩+|111⟩)/√2":
        reg = engine.build_ghz()
    elif initial_state == "W  (|001⟩+|010⟩+|100⟩)/√3":
        reg = engine.build_w()
    else:
        reg = QuantumRegister.from_zero()

    # Apply rune sequence
    if rune_chars:
        reg = engine.apply_rune_sequence(rune_chars, register=reg, target_qubit=0)

    # Bloch sphere (show=False required for Gradio)
    fig = BlochSphere.render_entanglement(reg, show=False)

    # Entanglement summary markdown
    summary = engine.entanglement_summary(reg)
    summary_md = (
        "| Qubit | Bloch vector | Entropy |\n"
        "|---|---|---|\n"
        f"| Huginn  | {tuple(round(v,3) for v in reg.bloch_vector(0))} | {summary['Huginn_entropy']:.4f} |\n"
        f"| Muninn  | {tuple(round(v,3) for v in reg.bloch_vector(1))} | {summary['Muninn_entropy']:.4f} |\n"
        f"| Metin   | {tuple(round(v,3) for v in reg.bloch_vector(2))} | {summary['Metin_entropy']:.4f} |\n\n"
        f"**State:** `{summary['state']}`"
    )

    # Oracle interpretation
    oracle_text = oracle.interpret(reg, rune_chars or [])

    # Optional HF push
    if push_to_hf and HF_CIRCUIT_REPO and HF_TOKEN:
        try:
            path = bridge.push_circuit_result(reg, rune_chars or [], HF_CIRCUIT_REPO)
            oracle_text += f"\n\n*✓ Saved to HF dataset: `{path}`*"
        except Exception as exc:
            oracle_text += f"\n\n*✗ HF upload failed: {exc}*"

    return fig, summary_md, oracle_text


# ── Tab 2: Hexagram Encoder ───────────────────────────────────────────────────

def show_hexagram(number: int):
    hx  = HEXAGRAMS[int(number)]
    amp = encode_hexagram(int(number))
    q   = Qubit.from_hexagram(int(number))
    fig = BlochSphere.render_single(q, label=f"{hx.name_zh} #{number}", show=False)

    info_md = (
        f"## {hx.name_zh} — {hx.name_en} (#{number})\n\n"
        f"**Binary:** `{hx.binary}` (bottom → top lines)\n\n"
        f"**Complex amplitude:**\n"
        f"- α (real) = `{amp.real:.6f}`\n"
        f"- β (imag) = `{amp.imag:.6f}`\n\n"
        f"**Probabilities:** |0⟩ = `{amp.real**2:.4f}`,  |1⟩ = `{amp.imag**2:.4f}`\n\n"
        f"**Bloch vector:** `{tuple(round(v,4) for v in q.bloch_vector())}`"
    )
    return info_md, fig


# ── Tab 3: Sensor Upload ──────────────────────────────────────────────────────

def _sensor_info_to_md(info: dict) -> str:
    lines = []
    for k, v in info.items():
        if k not in ("path", "stub"):
            lines.append(f"- **{k}:** `{v}`")
    return "\n".join(lines)


def toggle_upload_source(source: str):
    local_vis = source == "Local Upload"
    hf_vis    = source == "HuggingFace Dataset"
    return (
        gr.update(visible=local_vis),
        gr.update(visible=hf_vis),
        gr.update(visible=hf_vis),
    )


def process_sensor(
    upload_source: str,
    local_file,
    hf_repo: str,
    hf_filename: str,
):
    # Resolve path
    if upload_source == "Local Upload":
        if local_file is None:
            return "No file uploaded.", None
        path = local_file if isinstance(local_file, str) else local_file.name
    else:
        if not hf_repo or not hf_filename:
            return "Provide HF Dataset Repo ID and filename.", None
        try:
            path = bridge.pull_file(repo_id=hf_repo.strip(), filename=hf_filename.strip())
        except Exception as exc:
            return f"❌ Failed to download from HF: {exc}", None

    suffix = Path(path).suffix.lower()
    try:
        if suffix in LIDAR_FORMATS:
            sensor  = LiDARInput()
            info    = sensor.load(path)
            metin_q = sensor.to_qubit_encoding(info)
        elif suffix in VIDEO_FORMATS:
            sensor  = DashcamInput()
            info    = sensor.load_video(path)
            metin_q = sensor.to_qubit_encoding(info)
        else:
            return f"❌ Unsupported file type: `{suffix}`", None
    except Exception as exc:
        return f"❌ Sensor load error: {exc}", None

    fig = BlochSphere.render_single(metin_q, label="Metin (sensor)", show=False)

    result_md = (
        f"### {Path(path).name}\n\n"
        + _sensor_info_to_md(info) +
        f"\n\n---\n**Metin qubit:** `{metin_q}`\n\n"
        f"**Bloch vector:** `{tuple(round(v,4) for v in metin_q.bloch_vector())}`\n\n"
        f"**P(|0⟩):** `{metin_q.probability_zero():.4f}`  "
        f"**P(|1⟩):** `{metin_q.probability_one():.4f}`"
    )
    return result_md, fig


# ── Gradio layout ─────────────────────────────────────────────────────────────

with gr.Blocks(title="ODIN QuantumLogicGate v2.0") as demo:
    gr.Markdown(
        "# ᚢ ODIN QuantumLogicGate v2.0\n"
        "Norse mythology quantum simulator — "
        "64 I Ching hexagrams × Elder Futhark 24 runes × Huginn-Muninn-Metin entanglement"
    )

    # ── Tab 1 ────────────────────────────────────────────────────────────────
    with gr.Tab("⚡ Rune Circuit"):
        gr.Markdown(
            "> Select runes in casting order, choose an initial state, then **Run Circuit**.\n"
            "> ⚠ Two-qubit runes (ᚷ Gebo, ᛇ Eihwaz, ᛉ Algiz, ᛖ Ehwaz) act on qubits 0→1."
        )
        with gr.Row():
            with gr.Column(scale=1):
                rune_selector = gr.CheckboxGroup(
                    choices=RUNE_CHOICES,
                    label="Elder Futhark Runes",
                    value=[],
                )
                state_selector = gr.Radio(
                    choices=["|000⟩  (ground state)", "GHZ  (|000⟩+|111⟩)/√2", "W  (|001⟩+|010⟩+|100⟩)/√3"],
                    value="|000⟩  (ground state)",
                    label="Initial Register State",
                )
                push_toggle = gr.Checkbox(
                    label="Push result to HuggingFace dataset (requires HF_CIRCUIT_REPO)",
                    value=False,
                )
                run_btn = gr.Button("▶ Run Circuit", variant="primary", size="lg")

            with gr.Column(scale=2):
                bloch_plot     = gr.Plot(label="Bloch Sphere — Entanglement View")
                summary_output = gr.Markdown(label="Entanglement Summary")
                oracle_output  = gr.Textbox(
                    label="🔮 Rune Oracle Interpretation",
                    lines=5,
                    interactive=False,
                )

        run_btn.click(
            fn=run_rune_circuit,
            inputs=[rune_selector, state_selector, push_toggle],
            outputs=[bloch_plot, summary_output, oracle_output],
        )

    # ── Tab 2 ────────────────────────────────────────────────────────────────
    with gr.Tab("☯ Hexagram Encoder"):
        gr.Markdown(
            "> Each of the 64 I Ching hexagrams encodes a qubit amplitude using the golden ratio φ=1.618."
        )
        with gr.Row():
            with gr.Column(scale=1):
                hex_slider = gr.Slider(
                    minimum=1, maximum=64, step=1, value=1,
                    label="Hexagram Number (King Wen sequence)",
                )
                hex_btn = gr.Button("Encode", variant="primary")

            with gr.Column(scale=2):
                hex_info  = gr.Markdown()
                hex_bloch = gr.Plot(label="Bloch Sphere — Hexagram Qubit")

        hex_btn.click(fn=show_hexagram, inputs=[hex_slider], outputs=[hex_info, hex_bloch])
        hex_slider.change(fn=show_hexagram, inputs=[hex_slider], outputs=[hex_info, hex_bloch])

    # ── Tab 3 ────────────────────────────────────────────────────────────────
    with gr.Tab("📡 Sensor Upload"):
        gr.Markdown(
            "> Load a LiDAR point cloud or dashcam video. The file metadata is encoded as "
            "> the **Metin** (matter) qubit state — the quantum sensor interface."
        )
        with gr.Row():
            with gr.Column(scale=1):
                upload_source = gr.Radio(
                    choices=["Local Upload", "HuggingFace Dataset"],
                    value="Local Upload",
                    label="File Source",
                )
                local_file = gr.File(
                    label="Upload LiDAR (.pcd .las .ply) or Video (.mp4 .avi .mov .mkv)",
                    file_types=[".pcd", ".las", ".ply", ".mp4", ".avi", ".mov", ".mkv"],
                    visible=True,
                )
                hf_repo_input = gr.Textbox(
                    label="HF Dataset Repo ID",
                    placeholder=HF_DATASET_REPO or "username/odin-sensor-data",
                    visible=False,
                )
                hf_filename_input = gr.Textbox(
                    label="Filename in repo (e.g. lidar/scan001.las)",
                    visible=False,
                )
                upload_source.change(
                    fn=toggle_upload_source,
                    inputs=[upload_source],
                    outputs=[local_file, hf_repo_input, hf_filename_input],
                )
                sensor_btn = gr.Button("⚙ Process Sensor File", variant="primary", size="lg")

            with gr.Column(scale=2):
                sensor_result = gr.Markdown()
                metin_bloch   = gr.Plot(label="Metin Qubit — Bloch Sphere")

        sensor_btn.click(
            fn=process_sensor,
            inputs=[upload_source, local_file, hf_repo_input, hf_filename_input],
            outputs=[sensor_result, metin_bloch],
        )

    gr.Markdown(
        "---\n"
        "*ODIN QuantumLogicGate v2.0 · Apache 2.0 · "
        "[github.com/mawangsung/cicada](https://github.com/mawangsung/cicada)*"
    )


if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Soft(),
    )
