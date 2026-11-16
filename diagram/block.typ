#import "lib.typ": *

// One of the 44 joint blocks. Two residual streams, video on top and text below, that meet in the joint
// attention. Each stream branches towards the middle. The video modulation comes from the timestep
// embedding, the text modulation is a learned table that does not depend on the timestep.
#let joint-block() = cetz.canvas({
  import cetz.draw: *
  setup()

  let spine = 4
  let a-y = 1.4
  let one-y = 2.6
  let step = 0.62
  let attn-x = 7.2

  // modulation outputs in the order of the chunk in the code, with the x of the wire they feed
  let outs = (
    ("shift", "a", 4),
    ("scale", "a", 3),
    ("gate", "a", 10.9),
    ("shift", "f", 14),
    ("scale", "f", 13),
    ("gate", "f", 20.9),
  )

  // one stream, s = 1 for video and -1 for text
  let stream(s, tag, name, ffn-dim, token-sym) = {
    let y(v) = s * v
    io((-1.4, y(spine)), tag + "in", raw(name), side: "east")
    io((21.6, y(spine)), tag + "out", raw(name), side: "west")
    op((10.2, y(spine)), tag + "res-a", "+")
    op((20.2, y(spine)), tag + "res-f", "+")
    arrow(tag + "in.east", tag + "res-a.west")
    arrow(tag + "res-a.east", tag + "res-f.west")
    arrow(tag + "res-f.east", tag + "out.west")

    // attention branch, the attention itself is the box in the middle
    block((1.8, y(a-y)), tag + "norm-a", "LayerNorm")
    arrow((0.8, y(spine)), (0.8, y(a-y)), tag + "norm-a.west")
    op((3, y(a-y)), tag + "scale-a", "×")
    op((4, y(a-y)), tag + "shift-a", "+")
    arrow(tag + "norm-a.east", tag + "scale-a.west")
    arrow(tag + "scale-a.east", tag + "shift-a.west")
    arrow(tag + "shift-a.east", (attn-x - 0.75, y(a-y)))
    note((5.35, y(a-y)), token-sym)
    wire((attn-x + 0.75, y(a-y)), (10.2, y(a-y)))
    note((9.1, y(a-y)), token-sym)
    op((10.2, y(one-y)), tag + "gate-a", "×")
    arrow((10.2, y(a-y)), (10.2, y(one-y) - s * op-r))
    arrow((10.2, y(one-y) + s * op-r), (10.2, y(spine) - s * op-r))

    // FFN branch
    block((11.8, y(a-y)), tag + "norm-f", "LayerNorm")
    arrow((11.35, y(spine)), (11.35, y(a-y)), tag + "norm-f.west")
    op((13, y(a-y)), tag + "scale-f", "×")
    op((14, y(a-y)), tag + "shift-f", "+")
    arrow(tag + "norm-f.east", tag + "scale-f.west")
    arrow(tag + "scale-f.east", tag + "shift-f.west")
    block((15.2, y(a-y)), tag + "up", "Linear")
    arrow(tag + "shift-f.east", tag + "up.west")
    block((16.7, y(a-y)), tag + "act", [GELU tanh], h: 1.5)
    arrow(tag + "up.east", tag + "act.west")
    note((15.95, y(a-y)), ffn-dim)
    block((18.2, y(a-y)), tag + "down", "Linear")
    arrow(tag + "act.east", tag + "down.west")
    op((20.2, y(one-y)), tag + "gate-f", "×")
    arrow(tag + "down.east", (20.2, y(a-y)), (20.2, y(one-y) - s * op-r))
    arrow((20.2, y(one-y) + s * op-r), (20.2, y(spine) - s * op-r))
  }

  stream(1, "v-", "x_v", [32768], [N × 4096])
  stream(-1, "t-", "x_t", [16384], [L × 4096])

  // joint attention, detailed in its own figure
  stage((attn-x, 0), "attn", [Joint\ Attention], w: 1.5, h: 4.2)
  io((-1.4, 0), "pe", raw("P.E."), side: "east")
  arrow("pe.east", "attn.west")
  note((5.35, 0), [N × 128])

  // modulation of one stream, wires leave the chunk to the right and turn towards the stream
  let modulation(s, tag, chunk-x) = {
    let y(v) = s * v
    // first output on top in both streams, so the labels read in the order of the code
    let ys = range(6).map(i => s * 6.8 + 1.55 - i * step)
    block((chunk-x, y(6.8)), tag + "chunk", "chunk", h: 3.4)
    for (i, (kind, idx, x)) in outs.enumerate() {
      let y0 = ys.at(i)
      let lbl = [#kind#sub(idx)]
      note((chunk-x + block-w / 2 + 0.08, y0 + 0.02), lbl, side: "south-west")
      wire((chunk-x + block-w / 2, y0), (x, y0))
      let target = if kind == "gate" { y(one-y) } else if kind == "shift" { y(a-y + 0.28) } else { y(one-y + 0.2) }
      // the horizontals that reach beyond this wire's x and lie in its way
      let hops = (y(spine),)
      for (j, (_, _, xj)) in outs.enumerate() {
        let yj = ys.at(j)
        let between = (yj - y0) * (yj - target) < 0
        if j != i and xj > x and between { hops.push(yj) }
      }
      if kind == "scale" {
        op((x, y(one-y)), tag + "one-" + str(i), "+", r: 0.2)
        vwire(x, y0, target, hops: hops)
        arrow((x, y(one-y - 0.2)), (x, y(a-y + 0.28)))
        arrow((x - 0.8, y(one-y)), tag + "one-" + str(i) + ".west")
        note((x - 0.85, y(one-y)), "1", side: "east")
      } else if kind == "shift" {
        vwire(x, y0, target, hops: hops)
      } else {
        vwire(x, y0, target, hops: hops, head: false)
        arrow((x, y(one-y)), tag + "gate-" + (if idx == "a" { "a" } else { "f" }) + ".east")
      }
    }
  }

  // video: timestep embedding through SiLU and Linear
  io((-7.4, 6.8), "temb", raw("temb"), side: "east")
  block((-5.8, 6.8), "silu", "SiLU")
  block((-4.4, 6.8), "lin", "Linear")
  arrow("temb.east", "silu.west")
  arrow("silu.east", "lin.west")
  note((-3.55, 6.8), [6 × 4096])
  modulation(1, "v-", -2.6)
  arrow("lin.east", "v-chunk.west")

  // text: learned table, no input
  block((-4.4, -6.8), "table", [Parameter], h: 1.6)
  note((-3.55, -6.8), [6 × 4096])
  modulation(-1, "t-", -2.6)
  arrow("table.east", "t-chunk.west")
})
