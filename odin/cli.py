"""ODIN QuantumLogicGate v2.0 — CLI entry point.

Usage examples:
  python -m odin.cli encode --hexagram 42
  python -m odin.cli encode --rune ᚨ
  python -m odin.cli run-circuit --runes "ᚨ,ᚺ,ᛟ" --state ghz
  python -m odin.cli visualize --mode register --state ghz
  python -m odin.cli visualize --mode entanglement --state w
  python -m odin.cli sensor lidar --file path/to/cloud.pcd
  python -m odin.cli sensor dashcam --file path/to/video.mp4
"""

import argparse
import json
import sys


def cmd_encode(args):
    if args.hexagram is not None:
        from .mappings.hexagrams import encode_hexagram, HEXAGRAMS
        amp = encode_hexagram(args.hexagram)
        hx = HEXAGRAMS[args.hexagram]
        print(f"Hexagram {args.hexagram}: {hx.name_zh} / {hx.name_en}")
        print(f"  binary  : {hx.binary}")
        print(f"  alpha   : {amp.real:.6f}")
        print(f"  beta    : {amp.imag:.6f}")
        print(f"  |amp|   : {abs(amp):.6f}")
    elif args.rune is not None:
        from .mappings.dual_codec import DualEncoding
        enc = DualEncoding.from_rune(args.rune)
        print(repr(enc))
    else:
        print("Provide --hexagram N or --rune CHAR", file=sys.stderr)
        sys.exit(1)


def cmd_run_circuit(args):
    runes = [r.strip() for r in args.runes.split(",") if r.strip()]
    from .gates.entanglement import HuginnMuninnMetin
    engine = HuginnMuninnMetin()

    if args.state == "ghz":
        reg = engine.build_ghz()
        print("Initial state: GHZ")
    elif args.state == "w":
        reg = engine.build_w()
        print("Initial state: W")
    else:
        from .state.register import QuantumRegister
        reg = QuantumRegister.from_zero()
        print("Initial state: |000⟩")

    print(f"Applying rune sequence: {' '.join(runes)}")
    reg = engine.apply_rune_sequence(runes, register=reg, target_qubit=args.target_qubit)
    print(f"Result: {repr(reg)}")
    summary = engine.entanglement_summary(reg)
    print(json.dumps({k: v for k, v in summary.items() if k != "state"}, indent=2))


def cmd_visualize(args):
    from .gates.entanglement import HuginnMuninnMetin
    from .visualization.bloch import BlochSphere
    engine = HuginnMuninnMetin()

    if args.state == "ghz":
        reg = engine.build_ghz()
    elif args.state == "w":
        reg = engine.build_w()
    else:
        from .state.register import QuantumRegister
        reg = QuantumRegister.from_zero()

    if args.mode == "register":
        BlochSphere.render_register(reg, show=True)
    elif args.mode == "entanglement":
        BlochSphere.render_entanglement(reg, show=True)
    else:
        print(f"Unknown mode: {args.mode}", file=sys.stderr)
        sys.exit(1)


def cmd_sensor(args):
    if args.sensor_type == "lidar":
        from .sensor.lidar import LiDARInput
        sensor = LiDARInput()
        info = sensor.load(args.file)
        print(json.dumps(info, indent=2))
        qubit = sensor.to_qubit_encoding(info)
        print(f"Metin qubit: {qubit}")
    elif args.sensor_type == "dashcam":
        from .sensor.dashcam import DashcamInput
        sensor = DashcamInput()
        info = sensor.load_video(args.file)
        print(json.dumps(info, indent=2))
        qubit = sensor.to_qubit_encoding(info)
        print(f"Metin qubit: {qubit}")
    else:
        print(f"Unknown sensor type: {args.sensor_type}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        prog="odin",
        description="ODIN QuantumLogicGate v2.0 — Norse mythology quantum simulator",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # encode
    p_enc = sub.add_parser("encode", help="Encode a hexagram or rune as a qubit amplitude")
    p_enc.add_argument("--hexagram", type=int, metavar="N", help="Hexagram number 1-64")
    p_enc.add_argument("--rune", type=str, metavar="CHAR", help="Rune Unicode character")

    # run-circuit
    p_circ = sub.add_parser("run-circuit", help="Apply rune gate sequence to a register")
    p_circ.add_argument("--runes", type=str, required=True,
                        help="Comma-separated rune chars, e.g. 'ᚨ,ᚺ,ᛟ'")
    p_circ.add_argument("--state", type=str, default="zero",
                        choices=["zero", "ghz", "w"], help="Initial state")
    p_circ.add_argument("--target-qubit", type=int, default=0,
                        choices=[0, 1, 2], help="Primary qubit for single-qubit gates")

    # visualize
    p_vis = sub.add_parser("visualize", help="Open Bloch sphere visualization in browser")
    p_vis.add_argument("--mode", type=str, default="register",
                       choices=["register", "entanglement"])
    p_vis.add_argument("--state", type=str, default="ghz",
                       choices=["zero", "ghz", "w"])

    # sensor
    p_sen = sub.add_parser("sensor", help="Load sensor file and encode as Metin qubit")
    p_sen.add_argument("sensor_type", choices=["lidar", "dashcam"])
    p_sen.add_argument("--file", type=str, required=True, help="Path to sensor file")

    args = parser.parse_args()
    dispatch = {
        "encode": cmd_encode,
        "run-circuit": cmd_run_circuit,
        "visualize": cmd_visualize,
        "sensor": cmd_sensor,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
