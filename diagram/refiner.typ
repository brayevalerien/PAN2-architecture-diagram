#import "lib.typ": *

// One of the 4 token refiner blocks, a plain pre-norm transformer block with no modulation and no RoPE.
// Linears have no bias here.
#let refiner-block() = cetz.canvas({
  import cetz.draw: *
  setup()

  let a-y = 2.6
  let q-y = a-y + 1.3
  let v-y = a-y - 1.3

  // residual stream
  io((-1.4, 0), "in", raw("x_t"), side: "east")
  io((16, 0), "out", raw("x_t"), side: "west")
  op((7.2, 0), "res-a", "+")
  op((14.6, 0), "res-f", "+")
  arrow("in.east", "res-a.west")
  arrow("res-a.east", "res-f.west")
  arrow("res-f.east", "out.west")

  // ATTENTION
  block((0.8, a-y), "norm-a", "RMSNorm")
  arrow((0, 0), (0, a-y), "norm-a.west")
  block((2.2, a-y), "qkv", "Linear", h: 3.4)
  arrow("norm-a.east", "qkv.west")
  note((1.5, a-y), [12288])
  block((3.6, q-y), "q-norm", "RMSNorm", h: 1.1, size: 6.5pt)
  block((3.6, a-y), "k-norm", "RMSNorm", h: 1.1, size: 6.5pt)
  block((5, a-y), "attn", [Attention], h: 3.4, w: 0.75)
  for (y, l, n) in ((q-y, "Q", "q-norm"), (a-y, "K", "k-norm")) {
    arrow((2.2 + block-w / 2, y), n + ".west")
    arrow(n + ".east", (5 - 0.375, y))
    note((2.75, y), l)
  }
  arrow((2.2 + block-w / 2, v-y), (5 - 0.375, v-y))
  note((2.75, v-y), "V")
  note((5, v-y - 0.45), [32 heads × 128\ padding masked], side: "north")
  block((6.2, a-y), "o", "Linear")
  arrow("attn.east", "o.west")
  arrow("o.east", (7.2, a-y), "res-a.north")

  // FFN, SwiGLU: w2(w3(x) · SiLU(w1(x)))
  block((8.6, a-y), "norm-f", "RMSNorm")
  arrow((7.8, 0), (7.8, a-y), "norm-f.west")
  block((10.2, a-y + 0.9), "up", "Linear")
  block((10.2, a-y - 0.9), "gate", "Linear")
  wire("norm-f.east", (9.3, a-y))
  arrow((9.3, a-y + 0.9), "up.west")
  arrow((9.3, a-y - 0.9), "gate.west")
  wire((9.3, a-y + 0.9), (9.3, a-y - 0.9))
  note((10.2, a-y + 1.75), [w#sub[3]], side: "south")
  note((10.2, a-y - 1.75), [w#sub[1]], side: "north")
  block((11.5, a-y - 0.9), "silu", "SiLU")
  arrow("gate.east", "silu.west")
  op((12.8, a-y), "mul", "×")
  arrow("up.east", ((), "-|", "mul"), "mul.north")
  arrow("silu.east", ((), "-|", "mul"), "mul.south")
  block((13.8, a-y), "down", "Linear")
  arrow("mul.east", "down.west")
  note((11.7, a-y + 0.9), [16384], side: "south")
  note((13.8, a-y - 0.85), [w#sub[2]], side: "north")
  arrow("down.east", (14.6, a-y), "res-f.north")
})
