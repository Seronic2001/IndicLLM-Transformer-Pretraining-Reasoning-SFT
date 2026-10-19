"""Server of the demo: serves the built static site (demo/static/build_static output).

The page (index.html, app.js, lm.js, translit.js) runs every model in the visitor's
browser with ONNX Runtime Web, so this server only hands out files. It is the same
page on every host: the Hugging Face Space (sdk: gradio runs `python app.py`; FastAPI
and uvicorn come with Gradio) and Railway (Dockerfile from demo/build_space.py).

usage:
  python app.py                       serve ./site (or $DEMO_SITE) on $PORT (default 7860)
  python app.py --fetch-models URL    download site/models/<name>.onnx for every
                                      models/<name>.json that has no .onnx yet
"""
from __future__ import annotations

import argparse
import os
import shutil
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
SITE = Path(os.environ.get("DEMO_SITE", HERE / "site"))


def fetch_models(base_url: str, site: Path = SITE) -> None:
    """Fill in the ONNX files the site lists but does not hold (too big to upload to some hosts)."""
    for meta in sorted((site / "models").glob("*.json")):
        dst = meta.with_suffix(".onnx")
        if dst.exists():
            continue
        url = f"{base_url.rstrip('/')}/{dst.name}"
        print(f"  {dst.name} <- {url}", flush=True)
        tmp = dst.with_suffix(".part")
        with urllib.request.urlopen(url) as res, open(tmp, "wb") as f:
            shutil.copyfileobj(res, f)
        tmp.replace(dst)


def make_app(site: Path = SITE):
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles

    class Site(StaticFiles):
        """Models are cached for a week; everything else is revalidated, so a redeploy
        never leaves a visitor with a stale app.js."""

        async def get_response(self, path, scope):
            res = await super().get_response(path, scope)
            res.headers["Cache-Control"] = ("public, max-age=604800" if path.endswith(".onnx")
                                            else "no-cache")
            return res

    app = FastAPI(title="Hindi & Assamese mini-LMs", docs_url=None, redoc_url=None, openapi_url=None)
    app.mount("/", Site(directory=site, html=True), name="site")
    return app


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--fetch-models", metavar="URL", help="download missing ONNX models from URL and exit")
    args = parser.parse_args(argv)
    if args.fetch_models:
        fetch_models(args.fetch_models)
        return 0

    import uvicorn

    uvicorn.run(make_app(), host=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
                port=int(os.environ.get("PORT", os.environ.get("GRADIO_SERVER_PORT", 7860))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
