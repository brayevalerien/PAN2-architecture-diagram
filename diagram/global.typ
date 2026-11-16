#import "lib.typ": *

#let global-structure() = cetz.canvas({
  import cetz.draw: *
  setup()

  let main-y = -5
  let img-y = 0.2
  let mod-y = -3
  let txt-y = -6.8
  let pe-y = -8.8
  let cat-x = 11
  let xs = (14, 17, 22)
  let dots-x = 19.5
  let tail-x = 24.3
  let last-x = 25.6

  // INPUTS, the DiT sees the noisy latent, the first-frame latent and its mask stacked on channels
  io((0, 1.2), "in-z", raw("z_t"), side: "east")
  io((0, -0.5), "in-cond", raw("cond"), side: "east")
  io((0, -1.4), "in-mask", raw("mask"), side: "east")
  io((0, mod-y), "in-t", raw("timestep"), side: "east")
  io((0, pe-y), "in-grid", raw("grid"), side: "east")
  io((-9, txt-y), "in-cap", raw("caption"), side: "east")

  // video tokens
  op((1.8, -0.95), "cat1", "CAT")
  arrow("in-cond.east", ((), "-|", "cat1"), "cat1.north")
  arrow("in-mask.east", ((), "-|", "cat1"), "cat1.south")
  note((1.1, -0.5), [48])
  note((1.1, -1.4), [1])
  op((3.6, img-y), "cat2", "CAT")
  arrow("in-z.east", ((), "-|", "cat2"), "cat2.north")
  arrow("cat1.east", ((), "-|", "cat2"), "cat2.south")
  note((1.1, 1.2), [48])
  note((2.9, -0.95), [49])
  block((6.4, img-y), "patch", [Conv3d 1×2×2], h: 2.1)
  arrow("cat2.east", "patch.west")
  note((5.2, img-y), [97])
  wire("patch.east", (cat-x, img-y))
  note((8.8, img-y), [N × 4096])

  // timestep embedding, every block turns it into its own modulation
  block((2, mod-y), "t-sin", "Sinusoidal emb.", h: 2.1)
  block((3.6, mod-y), "t-mlp", "MLP Emb.")
  arrow("in-t.east", "t-sin.west")
  arrow("t-sin.east", "t-mlp.west")
  note(("t-sin.east", 50%, "t-mlp.west"), [256])
  wire("t-mlp.east", (last-x, mod-y))
  note((6, mod-y), [temb · 4096])

  // text: chat template, frozen Qwen3.5-9B, caption embedder, token refiner
  note((-9.8, txt-y - 0.45), [optionally rewritten by\ a language model\ 1536 tokens at most], side: "north")
  block((-8, txt-y), "tpl", "Chat template", h: 2.1)
  frozen((-6.4, txt-y), "qwen", [Qwen3.5-9B\ (frozen)], h: 2.4, w: 0.85)
  arrow("in-cap.east", "tpl.west")
  arrow("tpl.east", "qwen.west")
  note((-6.4, txt-y - 1.3), [hidden_states[-3], template\ tokens cropped], side: "north")
  block((-4.4, txt-y), "cap-norm", "RMSNorm")
  block((-3, txt-y), "cap-lin", "Linear")
  arrow("qwen.east", "cap-norm.west")
  arrow("cap-norm.east", "cap-lin.west")
  note((-5.35, txt-y), [L × 4096])
  stage((-1, txt-y), "rb1", [Refiner\ Block], w: 1.7)
  dots((0.6, txt-y))
  stage((2.2, txt-y), "rb2", [Refiner\ Block], w: 1.7)
  arrow("cap-lin.east", "rb1.west")
  arrow("rb1.east", (-0.1, txt-y))
  arrow((1.1, txt-y), "rb2.west")
  content((0.6, txt-y + 0.95), text(size: 15pt)[N = 4])
  op((4.6, txt-y), "ttype", "+")
  arrow("rb2.east", "ttype.west")
  io((4.6, txt-y - 1.2), "ttype-vec", raw("type_emb"), side: "center")
  arrow("ttype-vec.north", "ttype.south")
  note((8.2, txt-y), [L × 4096])

  // positions, only the video tokens are rotated
  block((2, pe-y), "rope", "3D RoPE")
  arrow("in-grid.east", "rope.west")
  wire("rope.east", (xs.last(), pe-y))
  note((6, pe-y), [P.E. · N × 128])
  note((6, pe-y - 0.05), [video tokens only], side: "north")

  // joint sequence, video first then text
  op((cat-x, main-y), "cat", "CAT")
  vwire(cat-x, img-y, main-y + op-r, hops: (mod-y,))
  arrow("ttype.east", ((), "-|", "cat"), "cat.south")

  // main stack
  for (i, x) in xs.enumerate() {
    stage((x, main-y), "b" + str(i), [Block])
    arrow((x, mod-y), "b" + str(i) + ".north")
    arrow((x, pe-y), "b" + str(i) + ".south")
  }
  dots((dots-x, main-y))
  arrow((dots-x, mod-y), (dots-x, main-y + 0.65))
  arrow((dots-x, pe-y), (dots-x, main-y - 0.65))
  arrow("cat.east", "b0.west")
  note(("cat.east", 50%, "b0.west"), [(N + L) × 4096])
  arrow("b0.east", "b1.west")
  arrow("b1.east", (dots-x - 0.6, main-y))
  arrow((dots-x + 0.6, main-y), "b2.west")
  content(((xs.first() + xs.last()) / 2, mod-y + 0.6), text(size: 15pt)[N = 44])

  // tail, only the video tokens reach the output head
  block((last-x, main-y), "last", "OutputHead", h: 1.9)
  arrow("b2.east", "last.west")
  arrow((last-x, mod-y), "last.north")
  note((tail-x + 0.6, main-y), [video])
  arrow((tail-x, main-y), (tail-x, main-y - 1.3))
  content((tail-x, main-y - 1.55), text(size: 16pt)[×])
  note((tail-x - 0.08, main-y - 0.6), [text], side: "east")
  block((27.4, main-y), "unpatch", "Unpatchify")
  arrow("last.east", "unpatch.west")
  note((26.5, main-y), [N × 192])
  io((29.4, main-y), "out", raw("v"), side: "west")
  arrow("unpatch.east", "out.west")
  note((28.5, main-y), [48 × T′HW])
})
