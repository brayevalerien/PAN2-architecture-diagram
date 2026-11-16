#import "lib.typ": *

// Joint attention: each stream has its own QKV, q/k RMSNorm and output projection, but q, k and v are
// concatenated (video first) and attend over all N + L tokens at once. Only the video q and k are rotated.
#let joint-attention() = cetz.canvas({
  import cetz.draw: *
  setup()

  let vy = 3
  let ty = -3
  let cq = (8, 1.2)
  let ck = (9.2, 0)
  let cv = (10.4, -1.2)

  // video stream
  io((0, vy), "xv", raw("x_v"), side: "east")
  block((1.2, vy), "vlin", "Linear")
  block((2.4, vy), "vchunk", "chunk", h: 3.4)
  arrow("xv.east", "vlin.west")
  arrow("vlin.east", "vchunk.west")
  note((1.8, vy), [12288])
  for (y, l) in ((vy + 1.4, "q"), (vy, "k"), (vy - 1.4, "v")) {
    note((2.45 + block-w / 2, y + 0.02), raw(l), side: "south-west")
  }
  block((3.9, vy + 1.4), "vnq", "RMSNorm", h: 1.25, size: 6.5pt)
  block((3.9, vy), "vnk", "RMSNorm", h: 1.25, size: 6.5pt)
  block((5.4, vy + 1.4), "vrq", "RoPE", h: 1)
  block((5.4, vy), "vrk", "RoPE", h: 1)
  arrow((2.4 + block-w / 2, vy + 1.4), "vnq.west")
  arrow((2.4 + block-w / 2, vy), "vnk.west")
  arrow("vnq.east", "vrq.west")
  arrow("vnk.east", "vrk.west")
  io((0, 6), "pe", raw("P.E."), side: "east")
  arrow("pe.east", (5.4, 6), "vrq.north")
  vwire(4.65, 6, vy - 0.8, hops: (vy + 1.4, vy), head: false)
  wire((4.65, vy - 0.8), (5.4, vy - 0.8))
  arrow((5.4, vy - 0.8), "vrk.south")
  note((2.9, 6), [N × 128], side: "south")

  // text stream, no RoPE
  io((0, ty), "xt", raw("x_t"), side: "east")
  block((1.2, ty), "tlin", "Linear")
  block((2.4, ty), "tchunk", "chunk", h: 3.4)
  arrow("xt.east", "tlin.west")
  arrow("tlin.east", "tchunk.west")
  note((1.8, ty), [12288])
  for (y, l) in ((ty + 1.4, "q"), (ty, "k"), (ty - 1.4, "v")) {
    note((2.45 + block-w / 2, y + 0.02), raw(l), side: "south-west")
  }
  block((3.9, ty + 1.4), "tnq", "RMSNorm", h: 1.25, size: 6.5pt)
  block((3.9, ty), "tnk", "RMSNorm", h: 1.25, size: 6.5pt)
  arrow((2.4 + block-w / 2, ty + 1.4), "tnq.west")
  arrow((2.4 + block-w / 2, ty), "tnk.west")
  note((6.4, ty - 0.1), [not rotated], side: "north")

  // concatenation, video enters from the north and text from the south
  op(cq, "catq", "CAT")
  op(ck, "catk", "CAT")
  op(cv, "catv", "CAT")

  wire("vrq.east", (cq.first(), vy + 1.4))
  vwire(cq.first(), vy + 1.4, cq.last() + op-r, hops: (vy, vy - 1.4))
  wire("vrk.east", (ck.first(), vy))
  vwire(ck.first(), vy, ck.last() + op-r, hops: (vy - 1.4, cq.last()))
  wire((2.4 + block-w / 2, vy - 1.4), (cv.first(), vy - 1.4))
  vwire(cv.first(), vy - 1.4, cv.last() + op-r, hops: (cq.last(), ck.last()))

  wire("tnq.east", (cq.first(), ty + 1.4))
  vwire(cq.first(), ty + 1.4, cq.last() - op-r, hops: ())
  wire("tnk.east", (ck.first(), ty))
  vwire(ck.first(), ty, ck.last() - op-r, hops: ())
  wire((2.4 + block-w / 2, ty - 1.4), (cv.first(), ty - 1.4))
  vwire(cv.first(), ty - 1.4, cv.last() - op-r, hops: ())

  // attention proper, one head at a time
  let core-y = 0.6
  op((12.4, core-y), "qk", "·")
  arrow("catq.east", ((), "-|", "qk"), "qk.north")
  arrow("catk.east", ((), "-|", "qk"), "qk.south")
  note((11.5, ck.last()), [Kᵀ], side: "south")
  op((13.6, core-y), "scale", "÷")
  arrow("qk.east", "scale.west")
  arrow((13.6, -0.5), "scale.south")
  note((13.6, -0.55), [√128], side: "north")
  block((14.8, core-y), "soft", "Softmax")
  arrow("scale.east", "soft.west")
  note((14.8, -0.45), [text padding\ masked], side: "north")
  op((16.4, core-y), "av", "·")
  arrow("soft.east", "av.west")
  arrow("catv.east", ((), "-|", "av"), "av.south")
  note((13.1, cv.last()), [V], side: "south")

  // split back into the two streams, each with its own output projection
  block((17.8, core-y), "split", "split", h: 2.6)
  arrow("av.east", "split.west")
  note((17.1, core-y), [(N + L)], side: "south")
  block((19.4, 1.7), "vo", "Linear")
  block((19.4, -0.5), "to", "Linear")
  arrow((17.8 + block-w / 2, 1.7), "vo.west")
  arrow((17.8 + block-w / 2, -0.5), "to.west")
  note((18.6, 1.7), [N])
  note((18.6, -0.5), [L])
  io((20.8, 1.7), "ov", raw("o_v"), side: "west")
  io((20.8, -0.5), "ot", raw("o_t"), side: "west")
  arrow("vo.east", "ov.west")
  arrow("to.east", "ot.west")
  note((14.8, 1.55), [32 heads × 128], side: "south")
})

// Interleaved RoPE as implemented: x cos + rotate_half(x) sin, where rotate_half maps every adjacent pair (a, b) to (−b, a)
#let rope-apply() = cetz.canvas({
  import cetz.draw: *
  setup()
  io((0, 0), "x", raw("x"), side: "east")
  block((1.5, -1.6), "pair", "Pair up", h: 1.4)
  arrow((0.7, 0), (0.7, -1.6), "pair.west")
  let out = 1.5 + block-w / 2
  op((3.1, -0.8), "neg", "−", r: 0.2)
  arrow((out, -2), (2.5, -2), (2.5, -0.8), "neg.west")
  op((4, -1.7), "stack", "stack", r: 0.34)
  arrow("neg.east", ((), "-|", "stack"), "stack.north")
  wire((out, -1.2), (2.2, -1.2))
  vwire(2.2, -1.2, -2.7, hops: (-2,), head: false)
  arrow((2.2, -2.7), (4, -2.7), "stack.south")
  note((out + 0.05, -1.2), [x#sub[2i]], side: "south-west")
  note((out + 0.05, -2), [x#sub[2i+1]], side: "south-west")
  block((5.2, -1.7), "flat", "Flatten", h: 1.2)
  arrow("stack.east", "flat.west")
  op((6.5, -1.7), "msin", "×")
  arrow("flat.east", "msin.west")
  op((6.5, 0), "mcos", "×")
  arrow("x.east", "mcos.west")
  op((7.7, 0), "add", "+")
  arrow("mcos.east", "add.west")
  arrow("msin.east", ((), "-|", "add"), "add.south")
  arrow("add.east", (8.7, 0))
  arrow((6.5, 1.1), "mcos.north")
  note((6.5, 1.15), [cos], side: "south")
  arrow((6.5, -2.8), "msin.south")
  note((6.5, -2.85), [sin], side: "north")
  note((3.6, 0), [float32 throughout], side: "south")
})

// Builds the cos and sin tables of the video tokens from their (t, h, w) grid index, one chain per axis
#let rope-embed() = cetz.canvas({
  import cetz.draw: *
  setup()
  io((0, 0), "ids", raw("index"), side: "east")
  op((1.5, 0), "scale", "×")
  arrow("ids.east", "scale.west")
  arrow((1.5, 1.1), "scale.north")
  note((1.5, 1.15), [16/24 on t, 1 on h, w], side: "south")
  op((3, 0), "mul", "×")
  arrow("scale.east", "mul.west")
  arrow((3, -1.1), "mul.south")
  note((3, -1.15), [ω#sub[j] = 256#super[−2j/d], j < d/2], side: "north")
  block((4.6, 0.9), "sin", "sin", h: 1)
  block((4.6, -0.9), "cos", "cos", h: 1)
  wire("mul.east", (4, 0))
  wire((4, 0.9), (4, -0.9))
  arrow((4, 0.9), "sin.west")
  arrow((4, -0.9), "cos.west")
  block((6.2, 0.9), "rs", "repeat ×2", h: 1.4, size: 6.5pt)
  block((6.2, -0.9), "rc", "repeat ×2", h: 1.4, size: 6.5pt)
  arrow("sin.east", "rs.west")
  arrow("cos.east", "rc.west")
  op((8, 0.9), "cats", "CAT")
  op((8, -0.9), "catc", "CAT")
  arrow("rs.east", ((), "-|", "cats"), "cats.west")
  arrow("rc.east", ((), "-|", "catc"), "catc.west")
  io((9.6, 0.9), "osin", raw("sin"), side: "west")
  io((9.6, -0.9), "ocos", raw("cos"), side: "west")
  arrow("cats.east", "osin.west")
  arrow("catc.east", "ocos.west")
  note((8, 0), [per axis, then CAT\ over t, h, w], side: "center")
})

#let position-table() = {
  set text(size: 7pt)
  table(
    columns: 4,
    stroke: none,
    align: center,
    inset: (x: 6pt, y: 3pt),
    table.hline(stroke: 0.6pt),
    [axis], [t], [h], [w],
    table.hline(stroke: 0.4pt),
    [head dims], [16], [56], [56],
    [index], [frame (4 video frames per latent frame after the first)], [row of the 2×2 patch grid], [column of the 2×2 patch grid],
    [position], [index × 16/24], [index], [index],
    table.hline(stroke: 0.6pt),
  )
}
