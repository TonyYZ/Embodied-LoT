/**
 * Embodied Language of Thought — Concept Demos
 * =============================================
 * Interactive demonstration of the embodied_lot.py model.
 * Each panel shows: concept structure · test scenes · activation scores.
 *
 * Activation scores were computed by embodied_lot.py and are hardcoded here.
 * The 9×9 receptive fields are reconstructed for display purposes.
 */

import { useState } from "react";

// ── Colour helpers ────────────────────────────────────────────────────────────
const lerp = (a, b, t) => a + (b - a) * t;
const fieldColour = (v) => {
  // 0 = indigo-950, 0.5 = slate-400, 1 = amber-200
  const r = Math.round(lerp(lerp(15, 148, v * 2), 254, Math.max(0, v * 2 - 1)));
  const g = Math.round(lerp(lerp(23, 163, v * 2), 240, Math.max(0, v * 2 - 1)));
  const b = Math.round(lerp(lerp(42, 187, v * 2), 138, Math.max(0, v * 2 - 1)));
  return `rgb(${r},${g},${b})`;
};
const barColour = (v) => {
  if (v >= 0.75) return "#f59e0b";
  if (v >= 0.55) return "#6366f1";
  return "#475569";
};

// ── Receptive field definitions (9×9) ────────────────────────────────────────
function makeField(patches, bg = 0.1) {
  const a = Array.from({ length: 9 }, () => Array(9).fill(bg));
  for (const [r0, r1, c0, c1, v] of patches) {
    for (let r = r0; r < r1; r++) for (let c = c0; c < c1; c++) a[r][c] = v;
  }
  return a;
}

const FIELD_DATA = {
  top:     makeField([[0, 3, 0, 9, 0.9]]),
  center:  makeField([[3, 6, 0, 9, 0.9]]),
  bottom:  makeField([[6, 9, 0, 9, 0.9]]),
  ring:    makeField([[0,2,0,9,0.9],[7,9,0,9,0.9],[0,9,0,2,0.9],[0,9,7,9,0.9]]),
  full:    makeField([[0,9,0,9,0.9]]),
  empty:   makeField([], 0.1),
  left:    makeField([[0,9,0,3,0.9]]),
  "obj+curl":  makeField([[0,9,0,9,0.85]]),   // object field (visual)
  "obj+flat":  makeField([[0,9,0,9,0.85]]),
  "noobj+curl":makeField([], 0.05),
  "noobj+flat":makeField([], 0.05),
  // approach motion fields: represent motion as overlay arrow colour
  "top-mot+ring": makeField([[0,2,0,9,0.9],[7,9,0,9,0.9],[0,9,0,2,0.9],[0,9,7,9,0.9]]),
  "bot-mot+ring": makeField([[0,2,0,9,0.9],[7,9,0,9,0.9],[0,9,0,2,0.9],[0,9,7,9,0.9]]),
  "top-mot":      makeField([[],], 0.5),
  "unif-mot":     makeField([[],], 0.5),
  "no-mot":       makeField([[],], 0.5),
};

// Motion overlays: which rows have motion (displayed as cyan tint)
const MOTION_ROWS = {
  "top-mot+ring": [0,1,2],
  "bot-mot+ring": [6,7,8],
  "top-mot":      [0,1,2],
  "unif-mot":     [0,1,2,3,4,5,6,7,8],
  "no-mot":       [],
};

// ── Components ─────────────────────────────────────────────────────────────────
function FieldGrid({ data, fieldKey, size = 72, label }) {
  const cells = data || FIELD_DATA[fieldKey] || makeField([]);
  const motionRows = MOTION_ROWS[fieldKey] || [];
  const px = Math.floor(size / 9);
  return (
    <div style={{ display:"flex", flexDirection:"column", alignItems:"center", gap:4 }}>
      <div style={{
        display: "grid", gridTemplateColumns: `repeat(9, ${px}px)`,
        gridTemplateRows: `repeat(9, ${px}px)`,
        border: "1px solid rgba(100,116,139,0.3)", borderRadius: 4,
      }}>
        {cells.map((row, r) =>
          row.map((v, c) => {
            const hasMotion = motionRows.includes(r);
            const base = fieldColour(v);
            return (
              <div key={`${r}-${c}`} style={{
                width: px, height: px,
                background: base,
                outline: hasMotion ? "1px solid rgba(34,211,238,0.6)" : "none",
                outlineOffset: -1,
              }} />
            );
          })
        )}
      </div>
      {label && (
        <span style={{ fontSize: 10, color: "#94a3b8", textAlign:"center",
                       maxWidth: size, lineHeight: 1.2 }}>{label}</span>
      )}
    </div>
  );
}

function ScoreBar({ score, maxScore = 1 }) {
  const pct = Math.round((score / maxScore) * 100);
  return (
    <div style={{ display:"flex", alignItems:"center", gap:6 }}>
      <div style={{
        flex: 1, height: 8, background: "rgba(71,85,105,0.25)", borderRadius: 4,
        overflow: "hidden",
      }}>
        <div style={{
          width: `${pct}%`, height: "100%",
          background: barColour(score),
          borderRadius: 4,
          transition: "width 0.4s ease",
        }} />
      </div>
      <span style={{ fontSize: 11, color: "#cbd5e1", minWidth: 32,
                     fontVariantNumeric: "tabular-nums" }}>
        {score.toFixed(3)}
      </span>
    </div>
  );
}

function ConceptTag({ label, color = "#6366f1" }) {
  return (
    <span style={{
      display:"inline-block", padding:"1px 8px", borderRadius:99,
      background: color + "22", border: `1px solid ${color}55`,
      fontSize: 11, color, fontFamily:"monospace", letterSpacing:"0.04em",
    }}>{label}</span>
  );
}

// ── Demo data ──────────────────────────────────────────────────────────────────
const DEMOS = [
  {
    id: "vocabulary",
    title: "Image-Schema Vocabulary",
    subtitle: "The eight trigrams as primitive spatial feature detectors",
    description:
      "Each ObjectNode encodes one Yijing trigram as a 3-band spatial template. " +
      "When placed as a sibling of a ReceptiveFieldNode inside a ParallelNode, it " +
      "scans the field and returns p ∈ [0,1] — the probability that the field matches " +
      "the expected yang-line distribution. Orientation (VERTICAL / HORIZONTAL) " +
      "selects which axis the bands run along.",
    graphLines: [
      "ParallelNode('par')",
      "  ├── RetinaNode        ← field context",
      "  └── ObjectNode('sc', 震, VERTICAL)  ← scanner",
    ],
    concepts: [
      { name:"震 ABOVE",    tag:"震",  tagC:"#f59e0b",
        note:"(1,0,0)  top band yang — figure in upper zone",
        fields:["top","center","bottom","ring","full","left"],
        labels:["top","center","bottom","ring","full","left"],
        scores:{top:0.900, center:0.367, bottom:0.367, ring:0.515, full:0.367, left:0.544} },
      { name:"艮 BELOW",    tag:"艮",  tagC:"#f97316",
        note:"(0,0,1)  bottom band yang — figure in lower zone",
        fields:["top","center","bottom","ring","full","left"],
        labels:["top","center","bottom","ring","full","left"],
        scores:{top:0.367, center:0.367, bottom:0.900, ring:0.515, full:0.367, left:0.544} },
      { name:"坎 CENTER",   tag:"坎",  tagC:"#06b6d4",
        note:"(0,1,0)  middle band yang — figure in central zone",
        fields:["top","center","bottom","ring","full","left"],
        labels:["top","center","bottom","ring","full","left"],
        scores:{top:0.367, center:0.900, bottom:0.367, ring:0.317, full:0.367, left:0.544} },
      { name:"离 BOUNDARY", tag:"离",  tagC:"#a78bfa",
        note:"(1,0,1)  outer bands yang — boundary / periphery presence",
        fields:["top","center","bottom","ring","full","left"],
        labels:["top","center","bottom","ring","full","left"],
        scores:{top:0.633, center:0.100, bottom:0.633, ring:0.683, full:0.633, left:0.456} },
      { name:"乾 UNIVERSAL",tag:"乾",  tagC:"#34d399",
        note:"(1,1,1)  all bands yang — full / universal presence",
        fields:["top","center","bottom","ring","full","left"],
        labels:["top","center","bottom","ring","full","left"],
        scores:{top:0.367, center:0.367, bottom:0.367, ring:0.653, full:0.900, left:0.367} },
      { name:"兑 LEFT-H",   tag:"兑",  tagC:"#fb7185",
        note:"(1,1,0) HORIZONTAL — majority in left+center columns",
        fields:["top","center","bottom","ring","full","left"],
        labels:["top","center","bottom","ring","full","left"],
        scores:{top:0.456, center:0.456, bottom:0.456, ring:0.485, full:0.633, left:0.633} },
    ],
  },

  {
    id: "containment",
    title: "CONTAINMENT Schema",
    subtitle: "离-H ∥ 离-V — radial boundary detector",
    description:
      "CONTAINMENT requires a closed boundary around an interior. " +
      "离 (1,0,1) detects outer-band presence; combining 离 in both HORIZONTAL " +
      "(detecting left/right walls) and VERTICAL (detecting top/bottom walls) " +
      "as a ParallelNode gives peak activation when all four sides are present. " +
      "This distinguishes a closed container from a simple top-heavy or center figure.",
    graphLines: [
      "ParallelNode('par')",
      "  ├── RetinaNode",
      "  ├── ObjectNode('离H', 离, HORIZONTAL)",
      "  └── ObjectNode('离V', 离, VERTICAL)",
    ],
    singleConcept: true,
    fields: ["ring","box_inner","top","center","full","empty"],
    labels: ["outer ring","ring+center","top only","center only","uniform full","empty"],
    fieldKeys:["ring","box_inner","top","center","full","empty"],
    scores: { ring:0.683, box_inner:0.599, top:0.544, center:0.278, full:0.633, empty:0.367 },
  },

  {
    id: "approach",
    title: "APPROACH Schema",
    subtitle: "震動 ⤳ modulated by 离靜 — dynamic proximity detector",
    description:
      "A dynamic ObjectNode (is_dynamic=True) scans the motion channel of the field " +
      "rather than its static activation. 震動 (top-band motion expected) detects " +
      "figures moving into the upper zone. A 离靜 (boundary / proximity) conditional " +
      "node gates 震動 via a dashed edge: when the figure is near the periphery " +
      "(approaching), the motion scanner's response is amplified by gain=1.5. " +
      "Cyan outlines in the field grids indicate regions with motion.",
    graphLines: [
      "ParallelNode('par')",
      "  ├── RetinaNode",
      "  ├── ObjectNode('震動', 震, VERTICAL, is_dynamic=True)   ← motion scanner",
      "  └── ObjectNode('离靜', 离, VERTICAL)",
      "       ⤳  震動  (gain=1.5)   ← dashed: proximity gates motion",
    ],
    singleConcept: true,
    fieldKeys:["top-mot+ring","bot-mot+ring","top-mot","unif-mot","no-mot"],
    labels:["top motion\n+ ring","bottom motion\n+ ring","top motion\nonly","uniform\nmotion","no\nmotion"],
    scores:{
      "top-mot+ring":0.808,
      "bot-mot+ring":0.541,
      "top-mot":     0.717,
      "unif-mot":    0.450,
      "no-mot":      0.583,
    },
  },

  {
    id: "grasp",
    title: "GRASP Schema — Multi-modal",
    subtitle: "Vision × Proprioception: two field hubs composed into one concept",
    description:
      "GRASP involves both a visible object (visual modality) and a curled-hand posture " +
      "(proprioceptive modality). Two ParallelNode 'hubs' each carry their own " +
      "ReceptiveFieldNode and scanner; the root ParallelNode combines them. " +
      "A dashed edge from the proprioceptive scanner (震靜, top-heavy = fingers curled) " +
      "to the visual scanner (乾靜, universal presence = object in view) modulates " +
      "visual sensitivity: feeling the object boosts what you see.",
    graphLines: [
      "ParallelNode('root')",
      "  ├── ParallelNode('vis')",
      "  │     ├── RetinaNode          ← visual field",
      "  │     └── ObjectNode('乾', 乾, VERTICAL)   ← object detector",
      "  └── ParallelNode('pro')",
      "        ├── ProprioNode         ← proprioceptive field",
      "        └── ObjectNode('震', 震, VERTICAL)   ← finger-curl detector",
      "              ⤳  乾  (gain=1.5)   ← proprio gates vision",
    ],
    singleConcept: true,
    multiModal: true,
    scenes: [
      { key:"Object + curled fingers", vis:"obj+curl",  pro:"top",        visl:"Object visible", prol:"Fingers curled" },
      { key:"Object + flat hand",      vis:"obj+flat",  pro:"empty",      visl:"Object visible", prol:"Hand flat" },
      { key:"No object + curled",      vis:"noobj+curl",pro:"top",        visl:"No object",      prol:"Fingers curled" },
      { key:"No object + flat",        vis:"noobj+flat",pro:"empty",      visl:"No object",      prol:"Hand flat" },
    ],
    scores:{
      "Object + curled fingers":0.900,
      "Object + flat hand":    0.767,
      "No object + curled":    0.500,
      "No object + flat":      0.367,
    },
  },
];

// ── Main component ────────────────────────────────────────────────────────────
export default function App() {
  const [activeDemo, setActiveDemo] = useState(0);
  const [activeConcept, setActiveConcept] = useState(0);
  const demo = DEMOS[activeDemo];

  return (
    <div style={{
      minHeight:"100vh", background:"#0f172a", color:"#e2e8f0",
      fontFamily:"'Georgia', 'Times New Roman', serif",
      padding:"0 0 48px",
    }}>
      {/* Header */}
      <div style={{
        borderBottom:"1px solid rgba(100,116,139,0.2)",
        padding:"20px 32px 16px",
        background:"rgba(15,23,42,0.95)",
        position:"sticky", top:0, zIndex:10,
      }}>
        <div style={{ maxWidth:900, margin:"0 auto" }}>
          <div style={{ display:"flex", alignItems:"baseline", gap:12, marginBottom:6 }}>
            <h1 style={{ margin:0, fontSize:18, fontWeight:"normal",
                         letterSpacing:"0.02em", color:"#f1f5f9" }}>
              Embodied Language of Thought
            </h1>
            <span style={{ fontSize:12, color:"#64748b",
                           fontFamily:"monospace" }}>embodied_lot.py</span>
          </div>
          <p style={{ margin:"0 0 14px", fontSize:12, color:"#64748b", lineHeight:1.5 }}>
            Concept activation demos — each node is a lazy scanner that only fires
            when given a ReceptiveFieldNode sibling in a ParallelNode.
            Activation ∈ [0,1] is a probability, never sampled.
          </p>
          <div style={{ display:"flex", gap:8, flexWrap:"wrap" }}>
            {DEMOS.map((d, i) => (
              <button key={d.id} onClick={() => { setActiveDemo(i); setActiveConcept(0); }}
                style={{
                  padding:"4px 14px", borderRadius:6, border:"none", cursor:"pointer",
                  fontSize:12, fontFamily:"inherit",
                  background: i===activeDemo ? "#6366f1" : "rgba(100,116,139,0.15)",
                  color: i===activeDemo ? "#fff" : "#94a3b8",
                  transition:"all 0.2s",
                }}>
                {d.title.split(" ").slice(0,2).join(" ")}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Body */}
      <div style={{ maxWidth:900, margin:"0 auto", padding:"28px 32px 0" }}>
        {/* Demo title */}
        <div style={{ marginBottom:20 }}>
          <h2 style={{ margin:"0 0 4px", fontSize:22, fontWeight:"normal",
                       color:"#f8fafc", letterSpacing:"0.01em" }}>
            {demo.title}
          </h2>
          <p style={{ margin:"0 0 10px", fontSize:13, color:"#94a3b8",
                      fontStyle:"italic" }}>
            {demo.subtitle}
          </p>
          <p style={{ margin:"0 0 16px", fontSize:13, color:"#cbd5e1", lineHeight:1.65 }}>
            {demo.description}
          </p>

          {/* Graph structure */}
          <div style={{
            background:"rgba(30,41,59,0.6)", border:"1px solid rgba(100,116,139,0.2)",
            borderRadius:8, padding:"12px 16px", marginBottom:20,
          }}>
            <div style={{ fontSize:10, color:"#64748b", marginBottom:6,
                          textTransform:"uppercase", letterSpacing:"0.08em" }}>
              Graph structure
            </div>
            {demo.graphLines.map((l,i) => (
              <div key={i} style={{
                fontFamily:"monospace", fontSize:12, color:"#a5b4fc",
                lineHeight:1.7, whiteSpace:"pre",
              }}>{l}</div>
            ))}
          </div>
        </div>

        {/* Vocabulary: concept picker */}
        {demo.id === "vocabulary" && (
          <>
            <div style={{ display:"flex", gap:8, flexWrap:"wrap", marginBottom:20 }}>
              {demo.concepts.map((c,i) => (
                <button key={c.name} onClick={() => setActiveConcept(i)}
                  style={{
                    padding:"5px 12px", borderRadius:6, cursor:"pointer",
                    fontSize:12, fontFamily:"inherit",
                    background: i===activeConcept ? c.tagC+"33" : "rgba(30,41,59,0.5)",
                    border:`1px solid ${i===activeConcept ? c.tagC+"88" : "rgba(100,116,139,0.25)"}`,
                    color: i===activeConcept ? c.tagC : "#94a3b8",
                    transition:"all 0.2s",
                  }}>
                  {c.tag} {c.name.split(" ").slice(1).join(" ")}
                </button>
              ))}
            </div>
            <VocabPanel c={demo.concepts[activeConcept]} />
          </>
        )}

        {/* Single-concept demos */}
        {demo.singleConcept && !demo.multiModal && (
          <SinglePanel demo={demo} />
        )}

        {/* Multi-modal demo */}
        {demo.multiModal && (
          <MultiModalPanel demo={demo} />
        )}
      </div>
    </div>
  );
}

function VocabPanel({ c }) {
  return (
    <div>
      <div style={{ display:"flex", alignItems:"center", gap:12, marginBottom:18 }}>
        <span style={{ fontSize:36 }}>{c.tag}</span>
        <div>
          <div style={{ fontSize:15, color:"#f1f5f9", marginBottom:2 }}>{c.name}</div>
          <div style={{ fontSize:12, color:"#94a3b8", fontFamily:"monospace" }}>{c.note}</div>
        </div>
      </div>

      <div style={{ display:"flex", gap:16, flexWrap:"wrap", marginBottom:24 }}>
        {c.fields.map((fk, i) => (
          <FieldGrid key={fk} fieldKey={fk} size={72} label={c.labels[i]} />
        ))}
      </div>

      <div style={{
        background:"rgba(30,41,59,0.5)", borderRadius:8,
        border:"1px solid rgba(100,116,139,0.2)", padding:16,
      }}>
        <div style={{ fontSize:10, color:"#64748b", textTransform:"uppercase",
                      letterSpacing:"0.08em", marginBottom:12 }}>
          Activation scores
        </div>
        {Object.entries(c.scores).map(([k, v]) => (
          <div key={k} style={{ display:"grid", gridTemplateColumns:"110px 1fr",
                                 gap:8, alignItems:"center", marginBottom:8 }}>
            <span style={{ fontSize:11, color:"#94a3b8", textAlign:"right" }}>{k}</span>
            <ScoreBar score={v} />
          </div>
        ))}
      </div>

      <p style={{ fontSize:11, color:"#475569", marginTop:16, lineHeight:1.6,
                  fontStyle:"italic" }}>
        Peak selectivity = 0.900 on the matching scene; drops to 0.367 for orthogonal
        scenes. The uniform-full scene scores 0.367 (not 0.5) because 乾 (1,1,1) expects
        all bands bright — partial patterns incur similarity costs on the yin-band positions.
      </p>
    </div>
  );
}

function SinglePanel({ demo }) {
  const maxScore = Math.max(...Object.values(demo.scores));
  return (
    <div>
      <div style={{ display:"flex", gap:14, flexWrap:"wrap", marginBottom:22 }}>
        {demo.fieldKeys.map((fk, i) => (
          <FieldGrid key={fk} fieldKey={fk} size={72} label={demo.labels[i]} />
        ))}
      </div>
      <div style={{
        background:"rgba(30,41,59,0.5)", borderRadius:8,
        border:"1px solid rgba(100,116,139,0.2)", padding:16,
      }}>
        <div style={{ fontSize:10, color:"#64748b", textTransform:"uppercase",
                      letterSpacing:"0.08em", marginBottom:12 }}>
          Activation scores
        </div>
        {demo.labels.map((label, i) => {
          const key = demo.fieldKeys[i];
          const v = demo.scores[key] ?? demo.scores[label];
          return (
            <div key={key} style={{ display:"grid", gridTemplateColumns:"160px 1fr",
                                     gap:8, alignItems:"center", marginBottom:8 }}>
              <span style={{ fontSize:11, color:"#94a3b8", textAlign:"right",
                             whiteSpace:"pre-line" }}>{label}</span>
              <ScoreBar score={v ?? 0} />
            </div>
          );
        })}
      </div>
    </div>
  );
}

function MultiModalPanel({ demo }) {
  return (
    <div>
      {/* Legend */}
      <div style={{ display:"flex", gap:16, marginBottom:20, flexWrap:"wrap" }}>
        <div style={{ display:"flex", alignItems:"center", gap:6 }}>
          <div style={{ width:12, height:12, borderRadius:2, background:"#6366f1" }} />
          <span style={{ fontSize:12, color:"#94a3b8" }}>Visual field (RetinaNode)</span>
        </div>
        <div style={{ display:"flex", alignItems:"center", gap:6 }}>
          <div style={{ width:12, height:12, borderRadius:2, background:"#f97316" }} />
          <span style={{ fontSize:12, color:"#94a3b8" }}>Proprioceptive field (ProprioNode)</span>
        </div>
      </div>

      <div style={{ display:"flex", flexDirection:"column", gap:16 }}>
        {demo.scenes.map(scene => {
          const score = demo.scores[scene.key] ?? 0;
          return (
            <div key={scene.key} style={{
              background:"rgba(30,41,59,0.5)", borderRadius:8,
              border:"1px solid rgba(100,116,139,0.2)",
              padding:16, display:"flex", gap:20, alignItems:"center",
            }}>
              {/* Fields side by side */}
              <div style={{ display:"flex", gap:12, flexShrink:0 }}>
                <div>
                  <div style={{ fontSize:9, color:"#6366f1", textTransform:"uppercase",
                                letterSpacing:"0.08em", marginBottom:4 }}>Visual</div>
                  <FieldGrid fieldKey={scene.vis} size={63} />
                  <div style={{ fontSize:10, color:"#64748b", textAlign:"center",
                                marginTop:3 }}>{scene.visl}</div>
                </div>
                <div style={{ display:"flex", alignItems:"center", color:"#475569",
                              fontSize:16, paddingBottom:14 }}>×</div>
                <div>
                  <div style={{ fontSize:9, color:"#f97316", textTransform:"uppercase",
                                letterSpacing:"0.08em", marginBottom:4 }}>Proprio</div>
                  <FieldGrid fieldKey={scene.pro} size={63} />
                  <div style={{ fontSize:10, color:"#64748b", textAlign:"center",
                                marginTop:3 }}>{scene.prol}</div>
                </div>
              </div>

              {/* Score */}
              <div style={{ flex:1 }}>
                <div style={{ fontSize:13, color:"#e2e8f0", marginBottom:8 }}>
                  {scene.key}
                </div>
                <ScoreBar score={score} />
                <div style={{ fontSize:11, color:"#64748b", marginTop:6, lineHeight:1.5 }}>
                  {score >= 0.85 ? "Both modalities active and consistent — strong grasp signal" :
                   score >= 0.70 ? "Object visible; hand posture not optimal" :
                   score >= 0.45 ? "Proprioception active; no object in visual field" :
                                   "Neither modality active"}
                </div>
              </div>
            </div>
          );
        })}
      </div>

      <div style={{
        marginTop:20, padding:14,
        background:"rgba(99,102,241,0.08)",
        border:"1px solid rgba(99,102,241,0.2)", borderRadius:8,
        fontSize:12, color:"#a5b4fc", lineHeight:1.7,
      }}>
        <strong>Dashed-edge cycle semantics.</strong> The proprio→visual dashed edge
        (gain 1.5) takes effect on the <em>next</em> evaluation cycle, not the current one —
        matching the cognitive intuition that felt contact shapes what you
        attend to on your next look, not a retroactive reweighting of what you just saw.
        The scores shown here reflect a steady-state two-call evaluation.
      </div>
    </div>
  );
}
