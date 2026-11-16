// Poster that assembles every figure, laid out like the Flux.1 and MOSS diagrams.
#import "lib.typ": *
#import "global.typ": global-structure
#import "block.typ": joint-block
#import "attention.typ": *
#import "refiner.typ": refiner-block
#import "small.typ": *
#import "pipeline.typ": *

#set document(title: "PAN 2 architecture diagram", author: "Valérien")
#set page(width: auto, height: auto, margin: 1.5cm, fill: white)
#set text(font: font)
#set align(center)

#let fig(body, name, size: 15pt, caption: none) = box(align(center, stack(
  dir: ttb,
  spacing: 0.6cm,
  body,
  text(size: size, name),
  ..if caption != none { (text(size: 7pt, caption),) },
)))

#let row(..figs) = align(center + bottom, stack(dir: ltr, spacing: 2.2cm, ..figs.pos().map(f => align(bottom, f))))

#stack(
  dir: ttb,
  spacing: 2.4cm,
  fig(scale(135%, reflow: true, global-structure()), [PAN 2 global architecture], size: 30pt),
  row(
    fig(joint-block(), [Joint Block], caption: [one of the 44 blocks, video on top and text below, the attention is shared]),
    fig(refiner-block(), [Refiner Block], caption: [one of the 4 blocks that process the caption before the stack]),
  ),
  row(
    fig(joint-attention(), [Joint Attention], caption: [the video q and k are rotated, the text ones are not]),
    fig(stack(dir: ttb, spacing: 1.2cm, rope-apply(), rope-embed()), [RoPE], caption: [interleaved pairs, theta = 256]),
    fig(position-table(), [Position ids]),
  ),
  row(
    fig(mlp-emb(), [MLP Emb.]),
    fig(modulation(), [Video modulation], caption: [one per block, different weights in each]),
    fig(text-modulation(), [Text modulation], caption: [one per block, no timestep]),
    fig(last-layer(), [OutputHead]),
  ),
  row(
    fig(latent-space(), [Latent space]),
    fig(guidance(), [Sampling step], caption: [50 steps, shift 9, guidance 2.5]),
    fig(sigma-schedule(), [Noise levels]),
  ),
  align(left, text(size: 8pt)[
    References
    - source code: github.com/ifm-ai/PAN2 (official), huggingface.co/IFM/PAN2 (diffusers layout)
    - paper: Critique of World Model, arXiv 2507.05169
    - symbols: T′ latent frames, H and W latent grid, N = T′ · H/2 · W/2 video tokens, L caption tokens, P.E. is a (cos, sin) pair of N × 128 tables
  ]),
)
