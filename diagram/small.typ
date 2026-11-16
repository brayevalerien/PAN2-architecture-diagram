#import "lib.typ": *

// Timesteps are σ × 1000, the sinusoidal embedding is cos then sin
#let mlp-emb() = cetz.canvas({
  import cetz.draw: *
  setup()
  io((0, 0), "in", raw("timestep"), side: "east")
  block((1.4, 0), "sin", [Sinusoidal emb.], h: 2.1)
  block((2.9, 0), "l1", "Linear")
  block((4.2, 0), "act", "SiLU")
  block((5.5, 0), "l2", "Linear")
  io((6.8, 0), "out", raw("temb"), side: "west")
  arrow("in.east", "sin.west")
  arrow("sin.east", "l1.west")
  arrow("l1.east", "act.west")
  arrow("act.east", "l2.west")
  arrow("l2.east", "out.west")
  note(("sin.east", 50%, "l1.west"), [256])
  note((4.85, 0), [4096])
  note((6.15, 0), [4096])
})

// One per block, computed from temb with its own weights. Starts at zero and runs in float32
#let modulation() = cetz.canvas({
  import cetz.draw: *
  setup()
  io((0, 0), "in", raw("temb"), side: "east")
  block((1.3, 0), "s", "SiLU")
  block((2.6, 0), "l", "Linear")
  block((4.6, 0), "c", "chunk", h: 2.4)
  arrow("in.east", "s.west")
  arrow("s.east", "l.west")
  arrow("l.east", "c.west")
  note((3.5, 0), [6 × 4096])
  let names = ([shift#sub[a]], [scale#sub[a]], [gate#sub[a]], [shift#sub[f]], [scale#sub[f]], [gate#sub[f]])
  for (i, n) in names.enumerate() {
    let y = 1 - i * 0.4
    arrow((4.6 + block-w / 2, y), (5.7, y))
    note((5.75, y - 0.01), n, side: "west")
  }
})

// The text stream has no timestep input, its six vectors are plain parameters of each block
#let text-modulation() = cetz.canvas({
  import cetz.draw: *
  setup()
  block((0, 0), "p", "Parameter", h: 1.6)
  block((1.6, 0), "c", "chunk", h: 2.4)
  arrow("p.east", "c.west")
  note((0.8, 0), [6 × 4096])
  let names = ([shift#sub[a]], [scale#sub[a]], [gate#sub[a]], [shift#sub[f]], [scale#sub[f]], [gate#sub[f]])
  for (i, n) in names.enumerate() {
    let y = 1 - i * 0.4
    arrow((1.6 + block-w / 2, y), (2.7, y))
    note((2.75, y - 0.01), n, side: "west")
  }
})

// Modulated LayerNorm and a Linear back to 1 × 2 × 2 × 48 latent values per token
#let last-layer() = cetz.canvas({
  import cetz.draw: *
  setup()
  io((0, 0), "in", raw("video"), side: "east")
  block((1.3, 0), "norm", "LayerNorm")
  op((2.7, 0), "scale", "×")
  op((3.9, 0), "shift", "+")
  block((5.1, 0), "l", "Linear")
  io((6.5, 0), "out", raw("tokens"), side: "west")
  arrow("in.east", "norm.west")
  arrow("norm.east", "scale.west")
  arrow("scale.east", "shift.west")
  arrow("shift.east", "l.west")
  arrow("l.east", "out.west")
  note((0.55, 0), [4096])
  note((5.85, 0), [192])

  io((-1.3, 3), "temb", raw("temb"), side: "east")
  block((0, 3), "s", "SiLU", h: 1.2)
  block((1.2, 3), "m", "Linear")
  block((2.3, 3), "c", "chunk", h: 1.2)
  arrow("temb.east", "s.west")
  arrow("s.east", "m.west")
  arrow("m.east", "c.west")
  note((1.75, 3), [8192])
  op((2.7, 1.2), "one", "+", r: 0.2)
  wire((2.3 + block-w / 2, 3.3), (3.9, 3.3))
  arrow((3.9, 3.3), "shift.north")
  wire((2.3 + block-w / 2, 2.7), (2.7, 2.7))
  arrow((2.7, 2.7), "one.north")
  arrow("one.south", "scale.north")
  arrow((1.9, 1.2), "one.west")
  note((1.85, 1.2), "1", side: "east")
  note((3.95, 2.2), [shift], side: "west")
  note((2.75, 2.2), [scale], side: "west")
})
