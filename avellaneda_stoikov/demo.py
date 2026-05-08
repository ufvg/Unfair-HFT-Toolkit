from __future__ import annotations

import argparse
import json
import time

from .model import AvellanedaStoikovModel, AvellanedaStoikovParameters
from .simulator import SimulationConfig, iter_simulation


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Avellaneda-Stoikov demo runner.")
    parser.add_argument("--gui", action="store_true", help="Launch the Tkinter GUI.")
    parser.add_argument("--steps", type=int, default=250, help="Number of simulated steps.")
    parser.add_argument("--dt", type=float, default=0.25, help="Simulation timestep in seconds.")
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--sigma", type=float, default=0.015)
    parser.add_argument("--arrival-scale", type=float, default=1.25)
    parser.add_argument("--arrival-decay", type=float, default=4.0)
    parser.add_argument("--horizon", type=float, default=30.0)
    parser.add_argument("--tick-size", type=float, default=0.5)
    parser.add_argument("--order-size", type=float, default=0.01)
    parser.add_argument("--sleep", type=float, default=0.0, help="Delay between printed demo steps.")
    return parser


def launch_gui() -> None:
    from .gui.avellaneda_stoikov_gui import launch_gui as gui_launch

    gui_launch()


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.gui:
        launch_gui()
        return
    params = AvellanedaStoikovParameters(
        gamma=args.gamma,
        sigma=args.sigma,
        arrival_rate_scale=args.arrival_scale,
        arrival_rate_decay=args.arrival_decay,
        horizon_seconds=args.horizon,
        tick_size=args.tick_size,
        order_size=args.order_size,
    )
    _ = AvellanedaStoikovModel(params)
    config = SimulationConfig(steps=args.steps, dt_seconds=args.dt, sigma=args.sigma)
    for step in iter_simulation(params, config):
        print(
            json.dumps(
                {
                    "t": step.snapshot.timestamp,
                    "mid": step.snapshot.midprice,
                    "bid": step.bid_price,
                    "ask": step.ask_price,
                    "reservation": step.reservation_price,
                    "inventory": step.inventory,
                    "pnl": step.pnl,
                    "bid_fill": step.bid_fill,
                    "ask_fill": step.ask_fill,
                }
            )
        )
        if args.sleep > 0.0:
            time.sleep(args.sleep)


if __name__ == "__main__":
    main()
