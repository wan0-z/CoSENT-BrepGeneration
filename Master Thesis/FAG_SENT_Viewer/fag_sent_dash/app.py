from __future__ import annotations
import argparse
import sys
from pathlib import Path

from dash import Dash, dcc, html, Input, Output, State, ctx, no_update

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from fag_sent_dash.build_cache import select_jsons
    from fag_sent_dash.models import load_part_json
    from fag_sent_dash.sent import sample_sent
    from fag_sent_dash.step_cache import load_part_cache
    from fag_sent_dash.frame_brep import make_brep_figure
    from fag_sent_dash.frame_sent import make_sent_figure
    from fag_sent_dash.frame_generation import make_generation_figure
else:
    from .build_cache import select_jsons
    from .models import load_part_json
    from .sent import sample_sent
    from .step_cache import load_part_cache
    from .frame_brep import make_brep_figure
    from .frame_sent import make_sent_figure
    from .frame_generation import make_generation_figure


APP_DIR = Path(__file__).resolve().parent.parent
DEFAULT_JSON_DIR = APP_DIR.parent / "Generation_Viewer" / "data" / "graphs"
DEFAULT_CACHE_DIR = APP_DIR / "web_cache"


def make_app(json_dir: str, cache_dir: str, seed: int = 42, n: int = 50) -> Dash:
    selected_paths = select_jsons(json_dir, seed, n)
    if not selected_paths:
        raise RuntimeError(f"No JSON files found in {json_dir}")
    records = [load_part_json(p) for p in selected_paths]

    app = Dash(__name__)
    app.title = "FAG SENT Dash"

    app.layout = html.Div([
        html.Div([
            html.Button("Prev", id="prev-part", n_clicks=0, style={"marginRight": "8px"}),
            html.Button("Next", id="next-part", n_clicks=0),
            html.Span(id="part-label", style={"marginLeft": "18px", "fontWeight": "600"}),
            html.Span(f" seed={seed}, sampled={len(records)}", style={"marginLeft": "12px", "color": "#666"}),
        ], style={"height": "42px", "display": "flex", "alignItems": "center"}),
        dcc.Store(id="part-index", data=0),
        dcc.Store(id="gen-step", data=0),
        html.Div([
            html.Div([
                dcc.Graph(id="brep-graph", style={"height": "calc(100vh - 64px)"}, config={"scrollZoom": True})
            ], style={"width": "33.33%", "border": "1px solid #ddd", "boxSizing": "border-box"}),
            html.Div([
                dcc.Graph(id="sent-graph", style={"height": "calc(100vh - 64px)"})
            ], style={"width": "33.33%", "border": "1px solid #ddd", "boxSizing": "border-box"}),
            html.Div([
                html.Div([
                    html.Button("←", id="prev-gen", n_clicks=0, style={"marginRight": "6px"}),
                    html.Button("→", id="next-gen", n_clicks=0),
                    html.Span(id="gen-label", style={"marginLeft": "12px"}),
                ], style={"position": "absolute", "zIndex": 10, "right": "10px", "top": "8px", "background": "rgba(255,255,255,0.85)", "padding": "4px"}),
                dcc.Graph(id="generation-graph", style={"height": "calc(100vh - 64px)"}, config={"scrollZoom": True}),
            ], style={"width": "33.33%", "border": "1px solid #ddd", "boxSizing": "border-box", "position": "relative"}),
        ], style={"display": "flex", "width": "100%"}),
    ], style={"fontFamily": "Arial, sans-serif"})

    def current_payload(part_idx: int):
        part = records[part_idx]
        cache = load_part_cache(part, cache_dir)
        # SENT seed combines global seed and part index to be deterministic but different per part.
        sent = sample_sent(part, seed=seed * 1000003 + part_idx)
        return part, cache, sent

    @app.callback(
        Output("part-index", "data"),
        Output("gen-step", "data", allow_duplicate=True),
        Input("prev-part", "n_clicks"),
        Input("next-part", "n_clicks"),
        State("part-index", "data"),
        prevent_initial_call=True,
    )
    def change_part(_prev, _next, idx):
        trig = ctx.triggered_id
        idx = int(idx or 0)
        if trig == "next-part":
            idx = (idx + 1) % len(records)
        elif trig == "prev-part":
            idx = (idx - 1) % len(records)
        return idx, 0

    @app.callback(
        Output("gen-step", "data", allow_duplicate=True),
        Input("prev-gen", "n_clicks"),
        Input("next-gen", "n_clicks"),
        State("part-index", "data"),
        State("gen-step", "data"),
        prevent_initial_call=True,
    )
    def change_generation_step(_prev, _next, part_idx, step_idx):
        trig = ctx.triggered_id
        _, _, sent = current_payload(int(part_idx or 0))
        max_step = max(0, len(sent.alternating_sequence) - 1)
        step_idx = int(step_idx or 0)
        if trig == "next-gen":
            step_idx = min(step_idx + 1, max_step)
        elif trig == "prev-gen":
            step_idx = max(step_idx - 1, 0)
        return step_idx

    @app.callback(
        Output("brep-graph", "figure"),
        Output("sent-graph", "figure"),
        Output("generation-graph", "figure"),
        Output("part-label", "children"),
        Output("gen-label", "children"),
        Input("part-index", "data"),
        Input("gen-step", "data"),
    )
    def render(part_idx, gen_step):
        part_idx = int(part_idx or 0)
        gen_step = int(gen_step or 0)
        part, cache, sent = current_payload(part_idx)
        brep_fig = make_brep_figure(cache, title=f"BREP: {part.part_id}")
        sent_fig = make_sent_figure(part, sent, seed=seed + part_idx)
        gen_fig, gen_label = make_generation_figure(cache, sent, gen_step)
        part_label = f"Part {part_idx + 1}/{len(records)}: {part.json_path.name}"
        return brep_fig, sent_fig, gen_fig, part_label, gen_label

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-dir", default=str(DEFAULT_JSON_DIR))
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8056)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    app = make_app(args.json_dir, args.cache_dir, args.seed, args.n)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
