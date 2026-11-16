#import "lib.typ": *

// How the first frame becomes the condition channels, and how the denoised latents become a video
#let latent-space() = cetz.canvas({
  import cetz.draw: *
  setup()

  // encode, image-to-video only
  io((0, 0), "in", raw("first frame"), side: "east")
  block((2.6, 0), "crop", [Resize and crop], h: 2.1)
  frozen((5, 0), "enc", [Wan 2.2 VAE\ encoder], h: 2.1, w: 0.85)
  op((7.4, 0), "sub", "−")
  op((8.6, 0), "div", "÷")
  block((11, 0), "place", [Frame 0], h: 1.3)
  io((13, 0), "out", raw("cond"), side: "west")
  arrow("in.east", "crop.west")
  arrow("crop.east", "enc.west")
  arrow("enc.east", "sub.west")
  arrow("sub.east", "div.west")
  arrow("div.east", "place.west")
  arrow("place.east", "out.west")
  arrow((7.4, -1), "sub.south")
  arrow((8.6, -1), "div.south")
  note((7.4, -1.05), [μ], side: "north")
  note((8.6, -1.05), [σ], side: "north")
  note(("in.east", 50%, "crop.west"), [3 × 704 × 1248])
  note(("enc.east", 50%, "sub.west"), [mean])
  note((9.85, 0), [48 × 1 × 44 × 78])
  note((11, -1), [other frames are zero,\ mask is 1 on frame 0], side: "north")

  // decode
  let y = -3.8
  io((0, y), "z", raw("z_0"), side: "east")
  op((2.6, y), "std", "×")
  op((3.8, y), "mean", "+")
  frozen((6.2, y), "dec", [Wan 2.2 VAE\ decoder], h: 2.1, w: 0.85)
  io((9.6, y), "vid", raw("video"), side: "west")
  arrow("z.east", "std.west")
  arrow("std.east", "mean.west")
  arrow("mean.east", "dec.west")
  arrow("dec.east", "vid.west")
  arrow((2.6, y - 1), "std.south")
  arrow((3.8, y - 1), "mean.south")
  note((2.6, y - 1.05), [σ], side: "north")
  note((3.8, y - 1.05), [μ], side: "north")
  note((1.2, y), [48 × T′ × 44 × 78])
  note((8.1, y), [3 × (4T′ − 3) × 704 × 1248])
})

// One sampling step: classifier free guidance, the first latent frame of image-to-video skips it, then an Euler update
#let guidance() = cetz.canvas({
  import cetz.draw: *
  setup()

  io((0, 0), "x", raw("x_t"), side: "east")
  io((1.6, 2.2), "pos", raw("caption"), side: "south")
  io((1.6, -3.2), "neg", raw("negative"), side: "north")
  stage((2.8, 1), "dit-c", [PAN 2], w: 2, h: 1)
  stage((2.8, -1.4), "dit-u", [PAN 2], w: 2, h: 1)
  wire("x.east", (0.9, 0))
  wire((0.9, 1), (0.9, -1.4))
  arrow((0.9, 1), "dit-c.west")
  arrow((0.9, -1.4), "dit-u.west")
  arrow("pos.south", (1.6, 1.5))
  arrow("neg.north", (1.6, -2))

  op((5.4, 1), "sub", "−")
  arrow("dit-c.east", "sub.west")
  note((4.2, 1), [v#sub[pos]])
  note((4.2, -1.4), [v#sub[neg]])
  op((6.5, 1), "cfg", "×")
  arrow("sub.east", "cfg.west")
  arrow((6.5, 2.1), "cfg.north")
  note((6.5, 2.15), [s = 2.5], side: "south")
  op((7.6, -1.4), "add", "+")
  wire("dit-u.east", (5.4, -1.4))
  arrow((5.4, -1.4), "sub.south")
  arrow((5.4, -1.4), "add.west")
  arrow("cfg.east", ((), "-|", "add"), "add.north")

  // image-to-video: frame 0 takes v_pos as it is
  wire((4.75, 1), (4.75, 2.8), (9, 2.8))
  block((9, -1.4), "i2v", [i2v frame 0], h: 1.9, size: 6.5pt)
  arrow("add.east", "i2v.west")
  arrow((9, 2.8), "i2v.north")

  op((10.4, -1.4), "dt", "×")
  arrow("i2v.east", "dt.west")
  arrow((10.4, -2.5), "dt.south")
  note((10.4, -2.55), [σ#sub[i+1] − σ#sub[i]], side: "north")
  op((11.6, 0), "step", "+")
  arrow("dt.east", ((), "-|", "step"), "step.south")
  wire((0.9, 0), (0.9, 3.5), (11.6, 3.5))
  arrow((11.6, 3.5), "step.north")
  io((12.8, 0), "next", raw("x_t+1"), side: "west")
  arrow("step.east", "next.west")
})

// Noise levels of the sampler, shifted towards noise, and the timestep the model sees
#let sigma-schedule() = cetz.canvas({
  import cetz.draw: *
  setup()
  io((0, 0), "in", raw("i"), side: "east")
  block((1.6, 0), "lin", [1 − i / 50], h: 1.6)
  block((3.2, 0), "shift", [9σ / (1 + 8σ)], h: 2.1)
  op((4.8, 0), "k", "×")
  io((6.2, 0), "out", raw("t"), side: "west")
  arrow("in.east", "lin.west")
  arrow("lin.east", "shift.west")
  arrow("shift.east", "k.west")
  arrow("k.east", "out.west")
  arrow((4.8, -1.1), "k.south")
  note((4.8, -1.15), [1000], side: "north")
  note((2.4, 0), [σ])
  note((4, 0), [σ′])
})
