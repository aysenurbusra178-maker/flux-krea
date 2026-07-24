#!/usr/bin/env python3
"""Generate with FLUX-Krea using the private VIZORA prompt library."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import click

from vizora_bridge import PromptReceipt, VizoraPromptClient


QUALITY_KEYS = (
    "identity",
    "skin",
    "anatomy",
    "handsFeet",
    "hair",
    "lighting",
    "composition",
    "motionContinuity",
)


def _client(sidecar_url: str | None, token: str | None) -> VizoraPromptClient:
    return VizoraPromptClient(base_url=sidecar_url, token=token)


def _print_json(value: Any) -> None:
    click.echo(json.dumps(value, indent=2, ensure_ascii=False))


@click.group()
def cli() -> None:
    """VIZORA + FLUX-Krea private local workflow."""


@cli.command()
@click.option("--objective", "-p", required=True, help="Creative brief or image objective.")
@click.option("--width", default=1024, type=click.IntRange(256, 2048), show_default=True)
@click.option("--height", default=1280, type=click.IntRange(256, 2048), show_default=True)
@click.option("--guidance", default=4.5, type=click.FloatRange(0.0, 20.0), show_default=True)
@click.option("--num-steps", default=28, type=click.IntRange(1, 100), show_default=True)
@click.option("--seed", default=42, type=int, show_default=True)
@click.option("--output", "-o", default="output.png", type=click.Path(path_type=Path), show_default=True)
@click.option("--image-role", multiple=True, help="Repeatable reference-role instruction.")
@click.option("--preserve", multiple=True, help="Repeatable preservation instruction.")
@click.option("--composition", multiple=True, help="Repeatable composition instruction.")
@click.option("--pose", multiple=True, help="Repeatable subject and pose instruction.")
@click.option("--outfit", multiple=True, help="Repeatable outfit and material instruction.")
@click.option("--environment", multiple=True, help="Repeatable environment instruction.")
@click.option("--camera", multiple=True, help="Repeatable camera instruction.")
@click.option("--lighting", multiple=True, help="Repeatable lighting instruction.")
@click.option("--avoid", multiple=True, help="Repeatable negative or avoid instruction.")
@click.option("--auto-evaluate/--no-auto-evaluate", default=False, show_default=True)
@click.option("--identity-reference", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--sidecar-url", envvar="VIZORA_SIDECAR_URL")
@click.option("--token", envvar="VIZORA_SIDECAR_TOKEN", hide_input=True)
def generate(
    objective: str,
    width: int,
    height: int,
    guidance: float,
    num_steps: int,
    seed: int,
    output: Path,
    image_role: tuple[str, ...],
    preserve: tuple[str, ...],
    composition: tuple[str, ...],
    pose: tuple[str, ...],
    outfit: tuple[str, ...],
    environment: tuple[str, ...],
    camera: tuple[str, ...],
    lighting: tuple[str, ...],
    avoid: tuple[str, ...],
    auto_evaluate: bool,
    identity_reference: Path | None,
    sidecar_url: str | None,
    token: str | None,
) -> None:
    """Compile a library prompt, run the existing inference script and save a receipt."""
    client = _client(sidecar_url, token)
    receipt = client.compile_prompt(
        objective=objective,
        image_roles=list(image_role),
        preserve=list(preserve),
        composition=list(composition),
        subject_and_pose=list(pose),
        outfit_and_materials=list(outfit),
        environment=list(environment),
        camera=list(camera),
        lighting_and_color=list(lighting),
        avoid=list(avoid),
        metadata={
            "width": str(width),
            "height": str(height),
            "guidance": str(guidance),
            "steps": str(num_steps),
            "seed": str(seed),
        },
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    receipt_path = output.with_suffix(f"{output.suffix}.vizora.json")
    receipt.save(receipt_path)

    inference_script = Path(__file__).with_name("inference.py")
    command = [
        sys.executable,
        str(inference_script),
        "--prompt",
        receipt.positive_prompt,
        "--width",
        str(width),
        "--height",
        str(height),
        "--guidance",
        str(guidance),
        "--num-steps",
        str(num_steps),
        "--seed",
        str(seed),
        "--output",
        str(output),
    ]

    click.echo(f"Compiled prompt: {receipt.prompt_id}")
    click.echo(f"Receipt: {receipt_path}")
    subprocess.run(command, check=True)

    if auto_evaluate:
        result = client.evaluate_image(
            receipt,
            output,
            identity_reference_path=identity_reference,
            notes="Automatic local visual evaluation after FLUX-Krea generation.",
        )
        _print_json(result)
    else:
        click.echo(
            "Generation completed. Run the rate command or enable --auto-evaluate to feed quality results back into the library."
        )


@cli.command()
@click.option("--receipt", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--accepted/--rejected", default=True, show_default=True)
@click.option("--identity", type=click.FloatRange(0.0, 1.0))
@click.option("--skin", type=click.FloatRange(0.0, 1.0))
@click.option("--anatomy", type=click.FloatRange(0.0, 1.0))
@click.option("--hands-feet", type=click.FloatRange(0.0, 1.0))
@click.option("--hair", type=click.FloatRange(0.0, 1.0))
@click.option("--lighting", type=click.FloatRange(0.0, 1.0))
@click.option("--composition", type=click.FloatRange(0.0, 1.0))
@click.option("--motion-continuity", type=click.FloatRange(0.0, 1.0))
@click.option("--notes", default="")
@click.option("--output-path", type=click.Path(path_type=Path))
@click.option("--sidecar-url", envvar="VIZORA_SIDECAR_URL")
@click.option("--token", envvar="VIZORA_SIDECAR_TOKEN", hide_input=True)
def rate(
    receipt: Path,
    accepted: bool,
    identity: float | None,
    skin: float | None,
    anatomy: float | None,
    hands_feet: float | None,
    hair: float | None,
    lighting: float | None,
    composition: float | None,
    motion_continuity: float | None,
    notes: str,
    output_path: Path | None,
    sidecar_url: str | None,
    token: str | None,
) -> None:
    """Send manual quality scores to the learning library."""
    prompt_receipt = PromptReceipt.load(receipt)
    scores = {
        key: value
        for key, value in {
            "identity": identity,
            "skin": skin,
            "anatomy": anatomy,
            "handsFeet": hands_feet,
            "hair": hair,
            "lighting": lighting,
            "composition": composition,
            "motionContinuity": motion_continuity,
        }.items()
        if value is not None
    }
    if not scores:
        raise click.UsageError("Provide at least one quality score between 0 and 1.")

    result = _client(sidecar_url, token).submit_feedback(
        prompt_receipt,
        accepted=accepted,
        scores=scores,
        notes=notes,
        output_path=str(output_path) if output_path else None,
    )
    _print_json(result)


@cli.command()
@click.option("--module-id", default="identity_skin_anatomy_master", show_default=True)
@click.option("--sidecar-url", envvar="VIZORA_SIDECAR_URL")
@click.option("--token", envvar="VIZORA_SIDECAR_TOKEN", hide_input=True)
def evolve(module_id: str, sidecar_url: str | None, token: str | None) -> None:
    """Create a candidate prompt version from accumulated feedback."""
    _print_json(_client(sidecar_url, token).evolve(module_id))


if __name__ == "__main__":
    cli()
