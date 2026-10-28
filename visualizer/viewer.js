    const payload = JSON.parse(document.getElementById("lot-data").textContent);
    // Node and board colours are theme tokens in viewer.css (--node-<type>, --surface, --unknown, --accent),
    // read at draw time so both themes follow.
    const nodeTypes = ["object", "out", "in", "parallel", "serial", "template", "field", "body", "input", "return", "plan"];
    const cssVar = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    const nodeColour = type => cssVar(`--node-${type}`) || cssVar("--soft") || "#7d8a82";

    const trigramPalette = [
      { name: "void", pattern: [0, 0, 0] },
      { name: "bottom", pattern: [1, 0, 0] },
      { name: "center", pattern: [0, 1, 0] },
      { name: "lower", pattern: [1, 1, 0] },
      { name: "top", pattern: [0, 0, 1] },
      { name: "periphery", pattern: [1, 0, 1] },
      { name: "upper", pattern: [0, 1, 1] },
      { name: "universal", pattern: [1, 1, 1] },
      { name: "neutral", pattern: [0.5, 0.5, 0.5] },
    ];

    // Add, remove, or rename built-in emoji icons here.
    // `key` is stored on the node; `emoji` is drawn; `label` appears in the picker.
    const iconEntries = [
      { key: "auto", emoji: "", label: "auto by node type" },
      { key: "eye", emoji: "👁️", label: "eye / input" },
      { key: "hand", emoji: "✋", label: "hand / output" },
      { key: "lens", emoji: "🔍", label: "scanner / lens" },
      { key: "gear", emoji: "⚙️", label: "process" },
      { key: "hub", emoji: "🕸️", label: "hub / parallel" },
      { key: "stack", emoji: "🔗", label: "stack / serial" },
      { key: "body", emoji: "🧍", label: "body" },
      { key: "plan", emoji: "📋", label: "plan" },
      { key: "star", emoji: "⭐", label: "goal" },
      { key: "key", emoji: "🔑", label: "gate" },
      { key: "clock", emoji: "⏱️", label: "temporal" },
      { key: "folder", emoji: "📁", label: "memory" },
      { key: "pencil", emoji: "✏️", label: "draw" },
      { key: "question", emoji: "❓", label: "unknown" },
      { key: "cross", emoji: "❌", label: "negation" },
      { key: "brain", emoji: "🧠", label: "concept" },
      { key: "robot", emoji: "🤖", label: "agent" },
      { key: "microscope", emoji: "🔬", label: "observer" },
      { key: "flask", emoji: "🧪", label: "experiment" },
      { key: "magnet", emoji: "🧲", label: "attractor" },
      { key: "compass", emoji: "🧭", label: "orientation" },
      { key: "target", emoji: "🎯", label: "target" },
      { key: "puzzle", emoji: "🧩", label: "composition" },
      { key: "lightning", emoji: "⚡", label: "activation" },
      { key: "seed", emoji: "🌱", label: "growth" },
      { key: "book", emoji: "📘", label: "memory / rule" },
      { key: "page", emoji: "📄", label: "information" },
      { key: "abacus", emoji: "🧮", label: "computation" },
      { key: "satellite", emoji: "🛰️", label: "signal" },
      { key: "joystick", emoji: "🕹️", label: "control" },
      { key: "spark", emoji: "✨", label: "imagined" },
      { key: "speaker", emoji: "🔊", label: "output / signal" },
      { key: "camera", emoji: "📷", label: "visual field" },
      { key: "map", emoji: "🗺️", label: "spatial map" },
      { key: "lock", emoji: "🔒", label: "inhibited" },
      { key: "unlock", emoji: "🔓", label: "released" },
      { key: "wave", emoji: "〰️", label: "motion" },
      { key: "dna", emoji: "🧬", label: "template" },
      { key: "atom", emoji: "⚛️", label: "primitive" },
      { key: "toolbox", emoji: "🧰", label: "tool" },
      { key: "beaker", emoji: "⚗️", label: "lab" },
      { key: "custom", emoji: "🖼️", label: "custom sprite" },
    ];

    const iconPalette = iconEntries.map(icon =>
      [icon.key, icon.emoji ? `${icon.emoji} ${icon.label}` : icon.label]
    );
    const emojiIconMap = Object.fromEntries(
      iconEntries.filter(icon => icon.emoji).map(icon => [icon.key, icon.emoji])
    );
    let frameIndex = 0;
    let current = structuredClone(payload.frames[0] || payload.base);
    let positions = new Map();
    let selected = null;
    let selectedKind = null;
    let selectedNodes = new Set();
    let timer = null;
    let physicsTimer = null;
    let draggedNode = null;
    let viewOffset = { x: 0, y: 0 };
    let viewScale = 1;
    let editMode = "select";
    let pendingEdge = null;
    let previewPoint = null;
    let marquee = null;
    let suppressNextCanvasClick = false;
    let labIconScale = 0.95;

    const svg = document.getElementById("graph");
    const edgeLayer = document.getElementById("edges");
    const nodeLayer = document.getElementById("nodes");
    const marqueeLayer = document.getElementById("marquee");
    const title = document.getElementById("title");
    const frameMeta = document.getElementById("frameMeta");
    const timeline = document.getElementById("timeline");

    function nodeById(id) {
      return current.nodes.find(n => n.id === id);
    }

    function roots() {
      return new Set(current.roots || []);
    }

    function graphSize() {
      const rect = svg.getBoundingClientRect();
      return { w: Math.max(rect.width, 600), h: Math.max(rect.height, 420) };
    }

    function layout() {
      const { w, h } = graphSize();
      const incoming = new Map(current.nodes.map(n => [n.id, 0]));
      const children = new Map(current.nodes.map(n => [n.id, []]));
      current.edges.filter(e => e.type === "solid").forEach(e => {
        incoming.set(e.target, (incoming.get(e.target) || 0) + 1);
        if (children.has(e.source)) children.get(e.source).push(e.target);
      });
      const rootIds = current.roots && current.roots.length
        ? current.roots
        : current.nodes.filter(n => (incoming.get(n.id) || 0) === 0).map(n => n.id);
      const levels = new Map();
      const queue = rootIds.map(id => [id, 0]);
      while (queue.length) {
        const [id, depth] = queue.shift();
        if (levels.has(id) && levels.get(id) <= depth) continue;
        levels.set(id, depth);
        (children.get(id) || []).forEach(child => queue.push([child, depth + 1]));
      }
      current.nodes.forEach((n, i) => {
        if (!levels.has(n.id)) levels.set(n.id, 1 + (i % 4));
      });
      const maxLevel = Math.max(...Array.from(levels.values()), 1);
      positions.clear();
      const levelStep = Math.min(150, Math.max(88, (h - 180) / Math.max(maxLevel, 1)));
      const totalHeight = maxLevel * levelStep;
      const yStart = Math.max(70, (h - totalHeight) / 2);
      // Lay out solid-edge trees by leaf order, centring every parent over
      // its children so sibling branches remain symmetric.
      const leafCount = (id, seen = new Set()) => {
        if (seen.has(id)) return 1;
        const kids = (children.get(id) || []).filter(child => !seen.has(child));
        if (!kids.length) return 1;
        const next = new Set(seen); next.add(id);
        return kids.reduce((sum, child) => sum + leafCount(child, next), 0);
      };
      const leafTotal = Math.max(1, rootIds.reduce((sum, id) => sum + leafCount(id), 0));
      const leafStep = leafTotal > 1 ? Math.max(74, Math.min(150, (w - 140) / (leafTotal - 1))) : 0;
      let cursor = leafTotal > 1 ? (w - leafStep * (leafTotal - 1)) / 2 : w / 2;
      const placed = new Set();
      const placeTree = (id, ancestry = new Set()) => {
        if (placed.has(id)) return positions.get(id).x;
        const level = levels.get(id) || 0;
        const next = new Set(ancestry); next.add(id);
        const kids = (children.get(id) || []).filter(child => !next.has(child));
        const x = kids.length
          ? kids.map(child => placeTree(child, next)).reduce((sum, value, _, all) => sum + value / all.length, 0)
          : cursor;
        if (!kids.length) cursor += leafStep;
        positions.set(id, { x, y: yStart + level * levelStep, vx: 0, vy: 0, level });
        placed.add(id);
        return x;
      };
      rootIds.forEach(id => placeTree(id));
      current.nodes.filter(node => !placed.has(node.id)).forEach(node => {
        const level = levels.get(node.id) || 0;
        positions.set(node.id, { x: cursor, y: yStart + level * levelStep, vx: 0, vy: 0, level });
        cursor += leafStep || 100;
      });
      startPhysics();
    }

    function resetLayoutOnce() {
      layout();
      if (!document.getElementById("floatToggle")?.checked) stopPhysics();
      render();
    }

    function applyFrame(index) {
      frameIndex = Math.max(0, Math.min(index, payload.frames.length - 1));
      const frame = payload.frames[frameIndex] || payload.base;
      const old = positions;
      const priorSelectedNodes = new Set(selectedNodes);
      const priorSelection = selectedKind === "node"
        ? { kind: "node", id: selected }
        : selectedKind === "nodes"
          ? { kind: "nodes", ids: [...selectedNodes] }
        : selectedKind === "edge" && current.edges[selected]
          ? { kind: "edge", edge: { ...current.edges[selected] } }
          : null;
      if (frame.engine) {                       // the engine's own state: never re-evaluated in the browser
        current = structuredClone(frame);
      } else if (document.getElementById("freeToggle")?.checked) {
        applyFrameState(frame);
      } else {
        current = structuredClone(frame);
      }
      const next = new Map();
      current.nodes.forEach(n => {
        if (old.has(n.id)) {
          const p = old.get(n.id);
          next.set(n.id, { ...p, vx: 0, vy: 0 });
        }
      });
      positions = next;
      if (positions.size !== current.nodes.length) layout();
      selectedNodes = new Set([...priorSelectedNodes].filter(id => nodeById(id)));
      if (priorSelection?.kind === "nodes") {
        selectedNodes = new Set(priorSelection.ids.filter(id => nodeById(id)));
        selected = null;
        selectedKind = selectedNodes.size > 1 ? "nodes" : null;
        if (selectedNodes.size === 1) {
          selected = [...selectedNodes][0];
          selectedKind = "node";
        }
      } else if (priorSelection?.kind === "node" && nodeById(priorSelection.id)) {
        selected = priorSelection.id;
        selectedKind = "node";
        selectedNodes = new Set([selected]);
      } else if (priorSelection?.kind === "edge") {
        const idx = current.edges.findIndex(edge =>
          edge.source === priorSelection.edge.source &&
          edge.target === priorSelection.edge.target &&
          edge.type === priorSelection.edge.type
        );
        selected = idx >= 0 ? idx : null;
        selectedKind = idx >= 0 ? "edge" : null;
        selectedNodes.clear();
      } else {
        selected = null;
        selectedKind = null;
        selectedNodes.clear();
      }
      render();
    }

    function applyFrameState(frame) {
      current.label = frame.label;
      current.graph_name = current.graph_name || frame.graph_name;
      const frameNodes = new Map((frame.nodes || []).map(node => [node.id, node]));
      current.nodes.forEach(node => {
        const source = frameNodes.get(node.id);
        if (!source) return;
        ["field", "position", "extent", "velocity", "subfield_shape"].forEach(key => {
          if (Object.prototype.hasOwnProperty.call(source, key)) {
            node[key] = structuredClone(source[key]);
          }
        });
      });
      evaluateEditedGraph(false);
    }

    function render() {
      title.textContent = payload.name || current.graph_name || "Embodied-LoT graph";
      frameMeta.textContent = `${current.label || "state"} | result ${fmt(current.result)} | nodes ${current.nodes.length} | edges ${current.edges.length}`;
      timeline.max = Math.max(payload.frames.length - 1, 0);
      timeline.value = frameIndex;
      applyViewport();
      renderModeButtons();
      renderSelectors();
      renderEdges();
      renderNodes();
      renderMarquee();
      renderDetails();
      renderStructureExport();
      renderEngineBoards();
      startPhysics();
    }

    function fmt(value) {
      return Number.isFinite(value) ? Number(value).toFixed(3) : "n/a";
    }

    function applyViewport() {
      const transform = `matrix(${viewScale} 0 0 ${viewScale} ${viewOffset.x} ${viewOffset.y})`;
      edgeLayer.setAttribute("transform", transform);
      nodeLayer.setAttribute("transform", transform);
      marqueeLayer.setAttribute("transform", transform);
    }

    function graphPoint(evt) {
      const p = pointer(evt);
      return { x: (p.x - viewOffset.x) / viewScale, y: (p.y - viewOffset.y) / viewScale };
    }

    function zoomGraph(evt) {
      if (evt.target.closest?.("aside")) return;
      evt.preventDefault();
      const p = pointer(evt);
      const before = { x: (p.x - viewOffset.x) / viewScale, y: (p.y - viewOffset.y) / viewScale };
      const factor = Math.exp(-evt.deltaY * 0.0012);
      const nextScale = Math.max(0.35, Math.min(2.8, viewScale * factor));
      viewScale = nextScale;
      viewOffset = {
        x: p.x - before.x * viewScale,
        y: p.y - before.y * viewScale,
      };
      applyViewport();
      renderEdges();
      renderNodes();
    }

    function setEditMode(mode) {
      editMode = mode;
      pendingEdge = null;
      previewPoint = null;
      renderModeButtons();
      renderEdges();
    }

    function renderModeButtons() {
      document.getElementById("modeSelectBtn")?.classList.toggle("primary", editMode === "select");
      document.getElementById("addBtn")?.classList.toggle("primary", editMode !== "select");
      svg.style.cursor = editMode === "node"
        ? "copy"
        : (editMode === "solid" || editMode === "dashed")
          ? "crosshair"
          : "grab";
    }

    function renderEdges() {
      edgeLayer.innerHTML = "";
      const showSolid = document.getElementById("solidToggle").checked;
      const showDashed = document.getElementById("dashedToggle").checked;
      const dashedEdges = current.edges.filter(edge => edge.type === "dashed");
      current.edges.forEach((e, idx) => {
        if (e.type === "solid" && !showSolid) return;
        if (e.type === "dashed" && !showDashed) return;
        const a = positions.get(e.source);
        const b = positions.get(e.target);
        if (!a || !b) return;
        const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
        g.classList.add("edge", e.type);
        if (selectedKind === "edge" && selected === idx) g.classList.add("selected");
        const path = svgEl("path", {
          d: e.type === "dashed"
            ? edgeDashedPath(a, b, e.source, e.target, dashedEdges.indexOf(e))
            : edgeStraightPath(a, b, e.source, e.target),
          "marker-end": e.type === "solid" ? "url(#arrowSolid)" : "url(#arrowDashed)",
        });
        path.addEventListener("click", evt => {
          evt.stopPropagation();
          selected = idx;
          selectedKind = "edge";
          selectedNodes.clear();
          render();
        });
        g.appendChild(path);
        edgeLayer.appendChild(g);
      });
      renderPendingEdge();
    }

    function edgeVisualRadius(nodeId) {
      const root = roots().has(nodeId);
      if (isLabViewMode()) {
        const iconSize = Math.round((root ? 50 : 44) * labIconScale);
        return iconSize / 2 + 14;
      }
      return (root ? 26 : 22) + 8;
    }

    function edgeStraightPath(a, b, sourceId = null, targetId = null) {
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const dist = Math.max(1, Math.sqrt(dx * dx + dy * dy));
      const ux = dx / dist;
      const uy = dy / dist;
      const sourcePad = sourceId ? edgeVisualRadius(sourceId) : 0;
      const targetPad = targetId ? edgeVisualRadius(targetId) + 12 : 0;
      const maxPad = Math.max(0, dist / 2 - 4);
      const s = Math.min(sourcePad, maxPad);
      const t = Math.min(targetPad, maxPad);
      const x1 = a.x + ux * s;
      const y1 = a.y + uy * s;
      const x2 = b.x - ux * t;
      const y2 = b.y - uy * t;
      return `M ${x1} ${y1} L ${x2} ${y2}`;
    }

    function edgeDashedPath(a, b, sourceId = null, targetId = null, index = 0) {
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const dist = Math.max(1, Math.sqrt(dx * dx + dy * dy));
      const ux = dx / dist;
      const uy = dy / dist;
      const sourcePad = sourceId ? edgeVisualRadius(sourceId) : 0;
      const targetPad = targetId ? edgeVisualRadius(targetId) + 12 : 0;
      const maxPad = Math.max(0, dist / 2 - 4);
      const s = Math.min(sourcePad, maxPad);
      const t = Math.min(targetPad, maxPad);
      const x1 = a.x + ux * s;
      const y1 = a.y + uy * s;
      const x2 = b.x - ux * t;
      const y2 = b.y - uy * t;
      // Route same-level conditional links beneath their nodes, away from the
      // solid hierarchy. Spread parallel dashed links into separate lanes.
      const lane = index % 3;
      if (Math.abs(y2 - y1) < 42) {
        const bend = 58 + lane * 24;
        return `M ${x1} ${y1} Q ${(x1 + x2) / 2} ${Math.max(y1, y2) + bend} ${x2} ${y2}`;
      }
      const bend = 34 + lane * 14;
      const nx = -uy;
      const ny = ux;
      return `M ${x1} ${y1} Q ${(x1 + x2) / 2 + nx * bend} ${(y1 + y2) / 2 + ny * bend} ${x2} ${y2}`;
    }

    function renderPendingEdge() {
      if (!pendingEdge || !previewPoint) return;
      const source = positions.get(pendingEdge.source);
      if (!source) return;
      const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
      g.classList.add("edge", "preview");
      const path = svgEl("path", {
        d: edgeStraightPath(source, previewPoint, pendingEdge.source, null),
        "marker-end": pendingEdge.type === "solid" ? "url(#arrowSolid)" : "url(#arrowDashed)",
      });
      g.appendChild(path);
      edgeLayer.appendChild(g);
    }

    function setNodeSelection(ids) {
      const valid = [...new Set(ids)].filter(id => nodeById(id));
      selectedNodes = new Set(valid);
      if (valid.length === 1) {
        selected = valid[0];
        selectedKind = "node";
      } else if (valid.length > 1) {
        selected = null;
        selectedKind = "nodes";
      } else {
        selected = null;
        selectedKind = null;
      }
    }

    function renderMarquee() {
      marqueeLayer.innerHTML = "";
      if (!marquee) return;
      const x = Math.min(marquee.start.x, marquee.current.x);
      const y = Math.min(marquee.start.y, marquee.current.y);
      const width = Math.abs(marquee.current.x - marquee.start.x);
      const height = Math.abs(marquee.current.y - marquee.start.y);
      marqueeLayer.appendChild(svgEl("rect", {
        class: "marquee-box",
        x,
        y,
        width,
        height,
      }));
    }

    function updatePendingPreview(evt) {
      if (!pendingEdge) return;
      previewPoint = graphPoint(evt);
      renderEdges();
    }

    function barColour(v) {
      if (v >= 0.75) return cssVar("--accent");
      if (v >= 0.45) return cssVar("--amber");
      return cssVar("--soft");
    }

    function isLabViewMode() {
      return document.body.classList.contains("labview");
    }

    function defaultIcon(node) {
      if (node.icon && node.icon !== "auto") return node.icon;
      if (node.type === "input") return "eye";
      if (node.type === "return") return "hand";
      if (node.type === "parallel") return "hub";
      if (node.type === "serial") return "stack";
      if (node.type === "body") return "body";
      if (node.type === "plan") return "plan";
      if (node.type === "template") return "dna";
      if (node.type === "field") return "map";
      if (node.type === "out") return "hand";
      if (node.type === "in") return "eye";
      if (node.dynamic) return "clock";
      return "lens";
    }

    function svgEl(name, attrs = {}) {
      const el = document.createElementNS("http://www.w3.org/2000/svg", name);
      Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, value));
      return el;
    }

    function renderEmojiIcon(parent, node, size = 26) {
      const iconKey = defaultIcon(node);
      const emoji = emojiIconMap[iconKey] || emojiIconMap.question;
      const fontFamily = "'Segoe UI Emoji', 'Apple Color Emoji', 'Noto Color Emoji', 'Twemoji Mozilla', sans-serif";
      const glyphSize = Math.round(size * 1.18);
      const emojiStyle = `font-family: ${fontFamily}; font-size: ${glyphSize}px; font-weight: 400; line-height: 1;`;
      const g = svgEl("g", { class: "xp-icon" });
      const shadow = svgEl("text", {
        class: "xp-emoji-shadow",
        x: 2.2,
        y: 1.6,
        "text-anchor": "middle",
        "dominant-baseline": "central",
        style: emojiStyle,
        opacity: 0.24,
        fill: "#000000",
      });
      shadow.textContent = emoji;
      const text = svgEl("text", {
        class: "xp-emoji",
        x: 0,
        y: -1.6,
        "text-anchor": "middle",
        "dominant-baseline": "central",
        style: emojiStyle,
      });
      text.textContent = emoji;
      g.appendChild(shadow);
      g.appendChild(text);
      parent.appendChild(g);
    }

    function renderXpIcon(parent, node, size = 26) {
      const iconKey = defaultIcon(node);
      const preloadedSprite = payload.icon_sprites?.[iconKey];
      if (node.iconImage || preloadedSprite) {
        const image = svgEl("image", {
          class: "xp-icon",
          x: -size / 2,
          y: -size / 2 - 3,
          width: size,
          height: size,
          href: node.iconImage || preloadedSprite,
          preserveAspectRatio: "xMidYMid meet",
        });
        parent.appendChild(image);
        return;
      }
      renderEmojiIcon(parent, node, size);
    }

    function renderNodes() {
      nodeLayer.innerHTML = "";
      const rootIds = roots();
      current.nodes.forEach(n => {
        const p = positions.get(n.id) || {x: 80, y: 80};
        const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
        g.classList.add("node");
        if ((selectedKind === "node" && selected === n.id) || selectedNodes.has(n.id)) g.classList.add("selected");
        g.setAttribute("transform", `translate(${p.x},${p.y})`);

        const r = rootIds.has(n.id) ? 26 : 22;
        const act = Math.max(0, Math.min(1, Number(n.activation) || 0));
        const attachNodeEvents = element => {
          element.addEventListener("pointerdown", evt => {
            if (editMode === "solid" || editMode === "dashed") {
              startEdgeDrag(n.id, evt);
              return;
            }
            dragStart(evt, n.id);
          });
          element.addEventListener("click", evt => {
            evt.stopPropagation();
            if (editMode === "node") return;
            if (editMode === "solid" || editMode === "dashed") {
              return;
            }
            setNodeSelection([n.id]);
            render();
          });
        };

        if (isLabViewMode()) {
          const iconSize = Math.round((rootIds.has(n.id) ? 50 : 44) * labIconScale);
          const hitRadius = Math.round(iconSize / 2 + 10);
          const hit = svgEl("circle", { class: "xp-hit", r: hitRadius });
          attachNodeEvents(hit);

          if ((selectedKind === "node" && selected === n.id) || selectedNodes.has(n.id)) {
            const boxSize = iconSize + 14;
            g.appendChild(svgEl("rect", {
              class: "xp-select-ring",
              x: -boxSize / 2,
              y: -boxSize / 2 - 4,
              width: boxSize,
              height: boxSize,
            }));
          }

          const meterWidth = iconSize + 10;
          const meterY = iconSize / 2 + 8;
          const meterTrack = svgEl("rect", {
            class: "xp-activation-track",
            x: -meterWidth / 2,
            y: meterY,
            width: meterWidth,
            height: 4,
            rx: 2,
            ry: 2,
          });
          const meterFill = svgEl("rect", {
            class: "xp-activation-fill",
            x: -meterWidth / 2 + 1,
            y: meterY + 1,
            width: Math.max(1, (meterWidth - 2) * act),
            height: 2,
            rx: 1,
            ry: 1,
            fill: act >= 0.75 ? "#00a000" : act >= 0.4 ? "#ffd400" : "#808080",
          });

          const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
          label.classList.add("xp-label");
          label.setAttribute("y", hitRadius + 18);
          label.textContent = shortId(n.id);

          g.appendChild(hit);
          renderXpIcon(g, n, iconSize);
          g.appendChild(meterTrack);
          g.appendChild(meterFill);
          g.appendChild(label);
          nodeLayer.appendChild(g);
          return;
        }

        const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        circle.setAttribute("r", r);
        // fill: desaturated base color, brightened by activation
        const base = nodeColour(n.type);
        circle.setAttribute("fill", base);
        circle.setAttribute("fill-opacity", (0.55 + act * 0.45).toFixed(2));
        attachNodeEvents(circle);

        const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
        text.textContent = shortId(n.id);
        const sub = document.createElementNS("http://www.w3.org/2000/svg", "text");
        sub.classList.add("sub");
        sub.setAttribute("y", r + 13);                    // below the circle, then the gauge
        sub.textContent = `${displayType(n)} ${fmt(n.activation)}`;

        const gaugeValue = Math.max(0, Math.min(1, Number(n.activation) || 0));
        const gaugeWidth = 66;
        const gaugeY = r + 24;
        const track = svgEl("rect", {
          class: "gate-track",
          x: -gaugeWidth / 2,
          y: gaugeY,
          width: gaugeWidth,
          height: 5,
          rx: 3,
          ry: 3,
        });
        const fill = svgEl("rect", {
          class: gaugeValue >= 0.75 ? "gate-fill high" : "gate-fill",
          x: -gaugeWidth / 2,
          y: gaugeY,
          width: gaugeWidth * gaugeValue,
          height: 5,
          rx: 3,
          ry: 3,
        });
        const gateNumber = svgEl("text", { class: "gate-number", x: gaugeWidth / 2 + 31, y: gaugeY + 5 });
        gateNumber.textContent = gaugeValue.toFixed(3);

        g.appendChild(circle);
        g.appendChild(text);
        g.appendChild(sub);
        g.appendChild(track);
        g.appendChild(fill);
        g.appendChild(gateNumber);
        nodeLayer.appendChild(g);
      });
    }

    function nodeRadius(nodeId) {
      if (isLabViewMode()) return roots().has(nodeId) ? 35 : 31;
      return roots().has(nodeId) ? 26 : 22;
    }

    function nodeAtGraphPoint(point, excludeId = null) {
      let best = null;
      let bestDist = Infinity;
      current.nodes.forEach(node => {
        if (node.id === excludeId) return;
        const p = positions.get(node.id);
        if (!p) return;
        const dx = point.x - p.x;
        const dy = point.y - p.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        const hitRadius = nodeRadius(node.id) + 10;
        if (dist <= hitRadius && dist < bestDist) {
          best = node;
          bestDist = dist;
        }
      });
      return best;
    }

    function startEdgeDrag(nodeId, evt) {
      if (editMode !== "solid" && editMode !== "dashed") {
        setNodeSelection([nodeId]);
        render();
        return;
      }
      evt.preventDefault();
      evt.stopPropagation();
      pendingEdge = { source: nodeId, type: editMode };
      previewPoint = graphPoint(evt);
      setNodeSelection([nodeId]);
      setStructureNote(`Drag to a target node and release to create a ${editMode} edge.`);
      render();

      function move(e) {
        previewPoint = graphPoint(e);
        renderEdges();
      }

      function up(e) {
        window.removeEventListener("pointermove", move);
        window.removeEventListener("pointerup", up);
        const target = nodeAtGraphPoint(graphPoint(e), nodeId);
        const completed = pendingEdge ? { ...pendingEdge, target: target?.id } : null;
        pendingEdge = null;
        previewPoint = null;
        if (!target || !completed?.target) {
          setStructureNote("Connection cancelled.");
          render();
          return;
        }
        if (edgeExists(completed.source, completed.target, completed.type)) {
          setNodeSelection([target.id]);
          setStructureNote(`${completed.type} edge already exists.`);
          render();
          return;
        }
        setNodeSelection([target.id]);
        addEdge(completed.type, completed.source, completed.target);
      }

      window.addEventListener("pointermove", move);
      window.addEventListener("pointerup", up);
    }

    function shortId(id) {
      return String(id).length > 9 ? String(id).slice(0, 8) + "." : String(id);
    }

    function dragStart(evt, id) {
      if (editMode !== "select") return;
      evt.preventDefault();
      pauseAutoLayoutForManualPose();
      if (!selectedNodes.has(id)) setNodeSelection([id]);
      const p = positions.get(id);
      const start = graphPoint(evt);
      const ox = p.x - start.x;
      const oy = p.y - start.y;
      const groupStarts = new Map([...selectedNodes]
        .filter(nodeId => positions.has(nodeId))
        .map(nodeId => [nodeId, { ...positions.get(nodeId) }]));
      function move(e) {
        const q = graphPoint(e);
        const nextX = q.x + ox;
        const nextY = q.y + oy;
        const dx = nextX - p.x;
        const dy = nextY - p.y;
        groupStarts.forEach((startPos, nodeId) => {
          positions.set(nodeId, { ...startPos, x: startPos.x + dx, y: startPos.y + dy, vx: 0, vy: 0 });
        });
        renderEdges();
        renderNodes();
      }
      function up() {
        draggedNode = null;
        window.removeEventListener("pointermove", move);
        window.removeEventListener("pointerup", up);
      }
      draggedNode = id;
      window.addEventListener("pointermove", move);
      window.addEventListener("pointerup", up);
      render();
    }

    function panStart(evt) {
      if (evt.target !== svg || editMode !== "select") return;
      evt.preventDefault();
      const start = pointer(evt);
      const ox = viewOffset.x;
      const oy = viewOffset.y;
      function move(e) {
        const q = pointer(e);
        viewOffset = { x: ox + q.x - start.x, y: oy + q.y - start.y };
        applyViewport();
      }
      function up() {
        window.removeEventListener("pointermove", move);
        window.removeEventListener("pointerup", up);
      }
      window.addEventListener("pointermove", move);
      window.addEventListener("pointerup", up);
    }

    function marqueeStart(evt) {
      if (evt.target !== svg || editMode !== "select") return;
      if (evt.altKey || evt.getModifierState?.("Space")) {
        panStart(evt);
        return;
      }
      evt.preventDefault();
      const start = graphPoint(evt);
      marquee = { start, current: start };
      function move(e) {
        marquee.current = graphPoint(e);
        renderMarquee();
      }
      function up(e) {
        window.removeEventListener("pointermove", move);
        window.removeEventListener("pointerup", up);
        const end = graphPoint(e);
        const x0 = Math.min(start.x, end.x);
        const x1 = Math.max(start.x, end.x);
        const y0 = Math.min(start.y, end.y);
        const y1 = Math.max(start.y, end.y);
        const draggedEnough = Math.abs(end.x - start.x) > 4 || Math.abs(end.y - start.y) > 4;
        marquee = null;
        if (!draggedEnough) {
          setNodeSelection([]);
          render();
          return;
        }
        suppressNextCanvasClick = true;
        const hits = current.nodes
          .filter(node => {
            const p = positions.get(node.id);
            return p && p.x >= x0 && p.x <= x1 && p.y >= y0 && p.y <= y1;
          })
          .map(node => node.id);
        setNodeSelection(hits);
        setStructureNote(hits.length ? `Selected ${hits.length} node${hits.length === 1 ? "" : "s"}.` : "No nodes in marquee.");
        render();
      }
      window.addEventListener("pointermove", move);
      window.addEventListener("pointerup", up);
      renderMarquee();
    }

    function startPhysics() {
      if (!document.getElementById("floatToggle")?.checked) return;
      if (physicsTimer) return;
      physicsTimer = requestAnimationFrame(physicsStep);
    }

    function stopPhysics() {
      if (physicsTimer) cancelAnimationFrame(physicsTimer);
      physicsTimer = null;
    }

    function pauseAutoLayoutForManualPose() {
      const toggle = document.getElementById("floatToggle");
      if (!toggle?.checked) return;
      toggle.checked = false;
      stopPhysics();
      setStructureNote("Auto layout paused so your manual node positions stay fixed.");
    }

    function physicsStep() {
      physicsTimer = null;
      if (!document.getElementById("floatToggle")?.checked) return;
      const { w, h } = graphSize();
      const nodes = current.nodes.map(n => n.id).filter(id => positions.has(id));
      const forces = new Map(nodes.map(id => [id, { x: 0, y: 0 }]));
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const a = positions.get(nodes[i]);
          const b = positions.get(nodes[j]);
          let dx = b.x - a.x;
          let dy = b.y - a.y;
          let d2 = dx * dx + dy * dy + 0.01;
          const d = Math.sqrt(d2);
          const push = 1450 / d2;
          dx /= d;
          dy /= d;
          forces.get(nodes[i]).x -= dx * push;
          forces.get(nodes[i]).y -= dy * push;
          forces.get(nodes[j]).x += dx * push;
          forces.get(nodes[j]).y += dy * push;
        }
      }
      current.edges.forEach(edge => {
        const a = positions.get(edge.source);
        const b = positions.get(edge.target);
        if (!a || !b) return;
        let dx = b.x - a.x;
        let dy = b.y - a.y;
        const d = Math.max(1, Math.sqrt(dx * dx + dy * dy));
        const target = edge.type === "solid" ? 180 : 230;
        const pull = (d - target) * (edge.type === "solid" ? 0.0026 : 0.0016);
        dx /= d;
        dy /= d;
        forces.get(edge.source).x += dx * pull;
        forces.get(edge.source).y += dy * pull;
        forces.get(edge.target).x -= dx * pull;
        forces.get(edge.target).y -= dy * pull;
      });
      nodes.forEach(id => {
        const p = positions.get(id);
        if (id === draggedNode) return;
        const level = Number.isFinite(p.level) ? p.level : 0;
        const maxLevel = Math.max(1, Math.max(...Array.from(positions.values()).map(v => v.level || 0)));
        const levelStep = Math.min(150, Math.max(88, (h - 180) / maxLevel));
        const guideY = Math.max(70, (h - maxLevel * levelStep) / 2) + level * levelStep;
        const f = forces.get(id);
        f.x += (w / 2 - p.x) * 0.00035;
        f.y += (h / 2 - p.y) * 0.00018;
        f.y += (guideY - p.y) * 0.00012;
        p.vx = (p.vx || 0) * 0.9 + f.x;
        p.vy = (p.vy || 0) * 0.9 + f.y;
        p.vx = Math.max(-4.5, Math.min(4.5, p.vx));
        p.vy = Math.max(-4.5, Math.min(4.5, p.vy));
        p.x = Math.max(38, Math.min(w - 38, p.x + p.vx));
        p.y = Math.max(38, Math.min(h - 38, p.y + p.vy));
      });
      renderEdges();
      renderNodes();
      physicsTimer = requestAnimationFrame(physicsStep);
    }

    function pointer(evt) {
      const pt = svg.createSVGPoint();
      pt.x = evt.clientX;
      pt.y = evt.clientY;
      const ctm = svg.getScreenCTM().inverse();
      const p = pt.matrixTransform(ctm);
      return { x: p.x, y: p.y };
    }

    function renderSelectors() {
      const source = document.getElementById("edgeSource");
      const target = document.getElementById("edgeTarget");
      if (!source || !target) return;
      const options = current.nodes.map(n => `<option value="${escapeHtml(n.id)}">${escapeHtml(n.id)}</option>`).join("");
      const oldSource = source.value;
      const oldTarget = target.value;
      source.innerHTML = options;
      target.innerHTML = options;
      if (nodeById(oldSource)) source.value = oldSource;
      if (nodeById(oldTarget)) target.value = oldTarget;
    }

    function displayType(node) {
      return node.type === "return" ? "output" : node.type;
    }

    function populateIconSelect() {
      const select = document.getElementById("nodeIcon");
      select.innerHTML = iconPalette
        .map(([value, label]) => `<option value="${escapeHtml(value)}">${escapeHtml(label)}</option>`)
        .join("");
    }

    function renderDetails() {
      const details = document.getElementById("details");
      const idInput = document.getElementById("nodeId");
      const typeInput = document.getElementById("nodeType");
      const iconInput = document.getElementById("nodeIcon");
      const iconFileInput = document.getElementById("nodeIconFile");
      const clearIconSpriteBtn = document.getElementById("clearIconSpriteBtn");
      const gateInput = document.getElementById("nodeGate");
      const actInput = document.getElementById("nodeActivation");
      const fieldPanel = document.getElementById("fieldPanel");
      const objectPatternPanel = document.getElementById("objectPatternPanel");
      const pixelPreviewRow = document.getElementById("pixelPreviewRow");
      if (selectedKind === "node") {
        const node = nodeById(selected);
        idInput.disabled = false;
        typeInput.disabled = false;
        iconInput.disabled = false;
        iconFileInput.disabled = false;
        clearIconSpriteBtn.disabled = !node.iconImage;
        gateInput.disabled = node.type === "return";
        actInput.disabled = false;
        idInput.value = node.id;
        typeInput.value = node.type || "object";
        iconInput.value = node.iconImage ? "custom" : (node.icon || "auto");
        iconFileInput.value = "";
        gateInput.value = fmt(node.type === "return" ? 1.0 : node.gate);
        actInput.value = fmt(node.activation);
        details.textContent = JSON.stringify({ ...node, display_type: displayType(node) }, null, 2);
        // Every node has a field view. Inputs preserve their observed field;
        // structural and imagined nodes receive a derived field on demand.
        if (node.type !== "input" || !node.field) {
          summarizeField(node, renderNodeWindow(node, 9, 9));
        }
        fieldPanel.classList.remove("hidden");
        pixelPreviewRow.classList.remove("hidden");
        syncFieldPanel(node);
        renderPixelPreview(fieldMatrix(node));
        if (node.type === "object") {
          objectPatternPanel.classList.remove("hidden");
          syncPatternEditor(node);
        } else {
          objectPatternPanel.classList.add("hidden");
        }
      } else if (selectedKind === "edge") {
        const edge = current.edges[selected];
        idInput.disabled = true;
        typeInput.disabled = true;
        iconInput.disabled = true;
        iconFileInput.disabled = true;
        clearIconSpriteBtn.disabled = true;
        gateInput.disabled = true;
        actInput.disabled = true;
        idInput.value = "";
        typeInput.value = "object";
        iconInput.value = "auto";
        iconFileInput.value = "";
        gateInput.value = "";
        actInput.value = "";
        details.textContent = JSON.stringify(edge, null, 2);
        fieldPanel.classList.add("hidden");
        objectPatternPanel.classList.add("hidden");
        pixelPreviewRow.classList.add("hidden");
      } else if (selectedKind === "nodes") {
        idInput.disabled = true;
        typeInput.disabled = true;
        iconInput.disabled = true;
        iconFileInput.disabled = true;
        clearIconSpriteBtn.disabled = true;
        gateInput.disabled = true;
        actInput.disabled = true;
        idInput.value = "";
        typeInput.value = "object";
        iconInput.value = "auto";
        iconFileInput.value = "";
        gateInput.value = "";
        actInput.value = "";
        details.textContent = JSON.stringify({
          selection: [...selectedNodes],
          count: selectedNodes.size,
        }, null, 2);
        fieldPanel.classList.add("hidden");
        objectPatternPanel.classList.add("hidden");
        pixelPreviewRow.classList.add("hidden");
      } else {
        idInput.disabled = true;
        typeInput.disabled = true;
        iconInput.disabled = true;
        iconFileInput.disabled = true;
        clearIconSpriteBtn.disabled = true;
        gateInput.disabled = true;
        actInput.disabled = true;
        idInput.value = "";
        typeInput.value = "object";
        iconInput.value = "auto";
        iconFileInput.value = "";
        gateInput.value = "";
        actInput.value = "";
        details.textContent = "Select a node or edge.";
        fieldPanel.classList.add("hidden");
        objectPatternPanel.classList.add("hidden");
        pixelPreviewRow.classList.add("hidden");
      }
    }

    function syncFieldPanel(node) {
      const isInput = node.type === "input";
      document.getElementById("fieldPanelTitle").textContent = isInput ? "Input Field" : "Output Field";
      document.getElementById("trigramPalette").classList.toggle("hidden", !isInput);
      document.getElementById("applyFieldBtn").classList.toggle("hidden", !isInput);
      document.getElementById("fieldEditor").readOnly = !isInput;
      document.getElementById("fieldNote").textContent = isInput
        ? "Editing this input refreshes browser-side scanner, dashed-gate, and output-window approximations for this frame."
        : "This output window is rendered from the non-window siblings parallel to O in the current browser snapshot.";
      syncFieldEditor(node);
    }

    function fieldColour(v) {
      // The board's three readings, as on the caterpillar plates: 0 known empty (surface), 0.5 unknown, 1 figure
      // (accent); values between blend the neighbouring two.
      const rgb = hex => {
        const h = hex.replace("#", "");
        const f = h.length === 3 ? h.split("").map(c => c + c).join("") : h;
        return [0, 2, 4].map(k => parseInt(f.slice(k, k + 2), 16));
      };
      const [lo, mid, hi] = ["--surface", "--unknown", "--accent"].map(name => rgb(cssVar(name) || "#888888"));
      const [a, b, t] = v <= 0.5 ? [lo, mid, v * 2] : [mid, hi, (v - 0.5) * 2];
      return `rgb(${a.map((x, k) => Math.round(x + (b[k] - x) * t)).join(",")})`;
    }

    function renderPixelPreview(values) {
      const preview = document.getElementById("pixelPreview");
      const h = values.length;
      const w = values[0].length;
      preview.innerHTML = "";
      preview.style.gridTemplateRows = `repeat(${h}, 1fr)`;
      preview.style.gridTemplateColumns = `repeat(${w}, 1fr)`;
      values.forEach(row => row.forEach(value => {
        const v = Math.max(0, Math.min(1, Number(value) || 0));
        const cell = document.createElement("span");
        cell.style.background = fieldColour(v);
        preview.appendChild(cell);
      }));
    }

    function syncPatternEditor(node) {
      document.getElementById("patternName").value = node.pattern_name || "";
      document.getElementById("patternEditor").value = Array.isArray(node.pattern) ? node.pattern.join(" ") : "";
      document.getElementById("orientationEditor").value = node.orientation || "vertical";
      document.getElementById("dynamicEditor").checked = Boolean(node.dynamic);
      document.getElementById("patternNote").textContent = "Palette buttons set this object node's pattern across replay frames.";
    }

    function parsePatternText(text) {
      const values = text.trim().split(/[\s,]+/).filter(Boolean).map(value => Number(value));
      if (!values.length || values.some(value => !Number.isFinite(value) || value < 0 || value > 1)) {
        throw new Error("Pattern must be numeric values in [0, 1].");
      }
      return values;
    }

    function applySelectedPattern() {
      if (selectedKind !== "node") return;
      const node = nodeById(selected);
      if (!node || node.type !== "object") return;
      try {
        const pattern = parsePatternText(document.getElementById("patternEditor").value);
        const patternName = document.getElementById("patternName").value.trim() || String(pattern);
        const orientation = document.getElementById("orientationEditor").value;
        const dynamic = document.getElementById("dynamicEditor").checked;
        applyPatternToNode(node, pattern, patternName, orientation, dynamic);
        payload.frames.forEach(frame => {
          const frameNode = frame.nodes.find(n => n.id === node.id && n.type === "object");
          if (frameNode) applyPatternToNode(frameNode, pattern, patternName, orientation, dynamic);
        });
        const baseNode = payload.base.nodes.find(n => n.id === node.id && n.type === "object");
        if (baseNode) applyPatternToNode(baseNode, pattern, patternName, orientation, dynamic);
        refreshReturnWindows(false);
        document.getElementById("patternNote").textContent = "Pattern updated across replay frames.";
        render();
      } catch (error) {
        document.getElementById("patternNote").textContent = error.message;
      }
    }

    function applyPatternToNode(node, pattern, patternName, orientation, dynamic) {
      node.pattern = [...pattern];
      node.pattern_name = patternName;
      node.name = patternName;
      node.orientation = orientation;
      node.dynamic = dynamic;
    }

    function syncFieldEditor(node) {
      const editor = document.getElementById("fieldEditor");
      const matrix = fieldMatrix(node);
      editor.value = matrix.map(row => row.map(v => Number(v).toFixed(2)).join(" ")).join("\n");
    }

    function fieldMatrix(node) {
      if (!node.field) node.field = {};
      if (Array.isArray(node.field.values)) return node.field.values;
      const shape = Array.isArray(node.field.shape) ? node.field.shape : [9, 9];
      const h = Math.max(1, Number(shape[0]) || 9);
      const w = Math.max(1, Number(shape[1]) || 9);
      const mean = Number.isFinite(node.field.mean) ? Number(node.field.mean) : 0;
      const values = Array.from({ length: h }, () => Array.from({ length: w }, () => mean));
      node.field.values = values;
      return values;
    }

    function matrixFilled(h, w, value = 0) {
      return Array.from({ length: h }, () => Array.from({ length: w }, () => value));
    }

    function summarizeField(node, values) {
      const flat = values.flat();
      node.field = node.field || {};
      node.field.values = values;
      node.field.shape = [values.length, values[0].length];
      node.field.mean = flat.reduce((a, b) => a + b, 0) / flat.length;
      node.field.min = Math.min(...flat);
      node.field.max = Math.max(...flat);
      return node.field.mean;
    }

    function parseFieldText(text) {
      const rows = text.trim().split(/\n+/).map(line =>
        line.trim().split(/[\s,]+/).filter(Boolean).map(value =>
          Math.max(0, Math.min(1, Number(value)))
        )
      ).filter(row => row.length);
      if (!rows.length || rows.some(row => row.some(value => !Number.isFinite(value)))) {
        throw new Error("Field must be numeric values in [0, 1].");
      }
      const width = rows[0].length;
      if (rows.some(row => row.length !== width)) {
        throw new Error("Every field row must have the same width.");
      }
      return rows;
    }

    function setSelectedField(values) {
      if (selectedKind !== "node") return;
      const node = nodeById(selected);
      if (!node || node.type !== "input") return;
      const flat = values.flat();
      node.field = node.field || {};
      node.field.values = values;
      node.field.shape = [values.length, values[0].length];
      node.field.mean = flat.reduce((a, b) => a + b, 0) / flat.length;
      node.field.min = Math.min(...flat);
      node.field.max = Math.max(...flat);
      node.activation = node.gate;
      scanInputSiblings(false);
      propagateDashed(false);
      refreshReturnWindows(false);
      document.getElementById("fieldNote").textContent = "Input snapshot changed; browser-side scanner and dashed-gate approximations were refreshed for this frame.";
      render();
    }

    function makeFieldFromPattern(pattern) {
      const node = selectedKind === "node" ? nodeById(selected) : null;
      const existing = node && node.type === "input" ? fieldMatrix(node) : null;
      const h = existing ? existing.length : 9;
      const w = existing ? existing[0].length : 9;
      const values = Array.from({ length: h }, () => Array.from({ length: w }, () => 0));
      const n = pattern.length;
      pattern.forEach((value, i) => {
        const bandFromTop = n - 1 - i;
        const r0 = Math.round(h * bandFromTop / n);
        const r1 = Math.round(h * (bandFromTop + 1) / n);
        for (let r = r0; r < r1; r++) values[r].fill(Number(value));
      });
      setSelectedField(values);
    }

    function setSelectedPatternFromPalette(item) {
      if (selectedKind !== "node") return;
      const node = nodeById(selected);
      if (!node || node.type !== "object") return;
      document.getElementById("patternName").value = item.name;
      document.getElementById("patternEditor").value = item.pattern.join(" ");
      applySelectedPattern();
    }

    function renderTrigramPaletteInto(palette, onPick) {
      palette.innerHTML = "";
      trigramPalette.forEach(item => {
        const button = document.createElement("button");
        button.type = "button";
        button.title = item.pattern.join(" ");
        const icon = document.createElement("span");
        icon.className = "tri-icon";
        [...item.pattern].reverse().forEach(value => {
          const line = document.createElement("span");
          line.className = "tri-line" + (Number(value) > 0 ? " on" : "");
          if (Number(value) === 0.5) line.style.opacity = "0.55";
          icon.appendChild(line);
        });
        const label = document.createElement("span");
        label.className = "tri-label";
        label.textContent = item.name;
        button.appendChild(icon);
        button.appendChild(label);
        button.addEventListener("click", () => onPick(item));
        palette.appendChild(button);
      });
    }

    function renderTrigramPalette() {
      renderTrigramPaletteInto(
        document.getElementById("trigramPalette"),
        item => makeFieldFromPattern(item.pattern)
      );
      renderTrigramPaletteInto(
        document.getElementById("patternTrigramPalette"),
        setSelectedPatternFromPalette
      );
    }

    function parentsOf(id) {
      return current.edges
        .filter(e => e.type === "solid" && e.target === id)
        .map(e => nodeById(e.source))
        .filter(Boolean);
    }

    function solidChildren(id) {
      return current.edges
        .filter(e => e.type === "solid" && e.source === id)
        .map(e => nodeById(e.target))
        .filter(Boolean);
    }

    function averageMatrices(matrices) {
      if (!matrices.length) return matrixFilled(9, 9, 0);
      const h = matrices[0].length;
      const w = matrices[0][0].length;
      const out = matrixFilled(h, w, 0);
      matrices.forEach(matrix => {
        for (let r = 0; r < h; r++) for (let c = 0; c < w; c++) out[r][c] += matrix[r][c] / matrices.length;
      });
      return out;
    }

    function meanChildFields(nodes, h = 9, w = 9) {
      if (!nodes.length) return matrixFilled(h, w, 0.5);
      const out = matrixFilled(h, w, 0);
      nodes.forEach(node => {
        const values = renderNodeWindow(node, h, w);
        for (let r = 0; r < h; r++) for (let c = 0; c < w; c++) out[r][c] += values[r][c] / nodes.length;
      });
      return out;
    }

    function renderObjectWindow(node, h = 9, w = 9) {
      const values = matrixFilled(h, w, 0);
      if (!Array.isArray(node.pattern)) return values;
      const observed = Boolean(fieldForObservedNode(node));
      const strength = Math.max(0, Math.min(1, Number(observed ? node.activation : node.gate) || 0));
      const n = node.pattern.length;
      node.pattern.forEach((bit, i) => {
        const patternValue = Number(bit) > 0 ? 1 : 0;
        const contrib = 0.5 + strength * (patternValue - 0.5);
        if (node.orientation === "horizontal") {
          const c0 = Math.round(w * i / n);
          const c1 = Math.round(w * (i + 1) / n);
          for (let r = 0; r < h; r++) for (let c = c0; c < c1; c++) values[r][c] = contrib;
        } else {
          const bandFromTop = n - 1 - i;
          const r0 = Math.round(h * bandFromTop / n);
          const r1 = Math.round(h * (bandFromTop + 1) / n);
          for (let r = r0; r < r1; r++) for (let c = 0; c < w; c++) values[r][c] = contrib;
        }
      });
      return values;
    }

    function renderNodeWindow(node, h = 9, w = 9) {
      if (node.type === "input") return fieldMatrix(node);
      if (node.type === "return") {
        const parent = parentsOf(node.id).find(Boolean);
        const siblings = parent ? solidChildren(parent.id).filter(child => child.type !== "input" && child.type !== "return") : [];
        return meanChildFields(siblings, h, w);
      }
      if (node.type === "object") return renderObjectWindow(node, h, w);
      if (node.type === "parallel") {
        const kids = solidChildren(node.id).filter(child => child.type !== "input" && child.type !== "return");
        return meanChildFields(kids, h, w);
      }
      if (node.type === "serial") {
        const kids = solidChildren(node.id).filter(child => child.type !== "input" && child.type !== "return");
        if (!kids.length) return matrixFilled(h, w, 0);
        const out = matrixFilled(h, w, 0);
        const horizontal = (node.split_orientation || "horizontal") === "horizontal";
        kids.forEach((child, i) => {
          const childValues = renderNodeWindow(child, h, w);
          const n = kids.length;
          if (horizontal) {
            const c0 = Math.round(w * i / n);
            const c1 = Math.round(w * (i + 1) / n);
            for (let r = 0; r < h; r++) for (let c = c0; c < c1; c++) out[r][c] = childValues[r][c];
          } else {
            const r0 = Math.round(h * (n - 1 - i) / n);
            const r1 = Math.round(h * (n - i) / n);
            for (let r = r0; r < r1; r++) for (let c = 0; c < w; c++) out[r][c] = childValues[r][c];
          }
        });
        return out;
      }
      if (node.type === "plan" || node.type === "body") {
        const kids = solidChildren(node.id).filter(child => child.type !== "input" && child.type !== "return");
        if (!kids.length) return matrixFilled(h, w, Number(node.activation) || 0);
        const content = kids[kids.length - 1];
        const template = kids[0];
        // A dynamic template supplies the content's layout direction, matching
        // the legacy tree renderer's orientation hand-off.
        if (template?.dynamic && content?.type === "serial" && template.orientation) {
          const previous = content.split_orientation;
          content.split_orientation = template.orientation;
          const values = renderNodeWindow(content, h, w);
          content.split_orientation = previous;
          return values;
        }
        return renderNodeWindow(content, h, w);
      }
      return matrixFilled(h, w, Number(node.activation) || 0);
    }

    function refreshReturnWindows(renderNow = true) {
      current.nodes.filter(n => n.type === "return").forEach(returnNode => {
        const parent = parentsOf(returnNode.id).find(Boolean);
        if (!parent) return;
        const siblings = solidChildren(parent.id).filter(child => child.type !== "input" && child.type !== "return");
        const values = meanChildFields(siblings, 9, 9);
        returnNode.activation = summarizeField(returnNode, values);
        returnNode.gate = 1.0;
      });
      if (renderNow) render();
    }

    function fieldForObservedNode(node) {
      for (const parent of parentsOf(node.id)) {
        if (parent.type !== "parallel") continue;
        const input = solidChildren(parent.id).find(child => child.type === "input" && child.field);
        if (input) return input;
      }
      return null;
    }

    function bandMeans(values, patternLength, orientation) {
      const h = values.length;
      const w = values[0].length;
      const means = [];
      for (let i = 0; i < patternLength; i++) {
        let sum = 0;
        let count = 0;
        if (orientation === "horizontal") {
          const c0 = Math.round(w * i / patternLength);
          const c1 = Math.round(w * (i + 1) / patternLength);
          for (let r = 0; r < h; r++) for (let c = c0; c < c1; c++) { sum += values[r][c]; count++; }
        } else {
          const bandFromTop = patternLength - 1 - i;
          const r0 = Math.round(h * bandFromTop / patternLength);
          const r1 = Math.round(h * (bandFromTop + 1) / patternLength);
          for (let r = r0; r < r1; r++) for (let c = 0; c < w; c++) { sum += values[r][c]; count++; }
        }
        means.push(count ? sum / count : 0);
      }
      return means;
    }

    function evaluateObjectInBrowser(node, values) {
      if (!Array.isArray(node.pattern)) return Number(node.activation) || 0;
      if (node.pattern.every(v => Number(v) === 0.5)) return 0.5 * (Number(node.gate) || 0);
      const means = bandMeans(values, node.pattern.length, node.orientation || "vertical");
      const sims = [];
      node.pattern.forEach((expected, i) => {
        if (Number(expected) > 0) sims.push(1 - Math.abs(means[i] - Number(expected)));
      });
      if (!sims.length) return 0;
      return Math.max(0, Math.min(1, (Number(node.gate) || 0) * (sims.reduce((a, b) => a + b, 0) / sims.length)));
    }

    function scanInputSiblings(renderNow = true) {
      current.nodes.filter(n => n.type === "object").forEach(node => {
        const input = fieldForObservedNode(node);
        if (!input) return;
        node.activation = evaluateObjectInBrowser(node, fieldMatrix(input));
      });
      if (renderNow) render();
    }

    function resetBrowserState() {
      current.nodes.forEach(node => {
        node.gate = 1.0;
        if (node.type === "input" || node.type === "return") {
          node.activation = 1.0;
        } else if (!fieldForObservedNode(node)) {
          node.activation = Number(node.gate) || 1.0;
        }
      });
    }

    function updateRootResultFromEditedGraph() {
      const outputs = current.nodes.filter(node => node.type === "return");
      if (outputs.length) {
        current.result = Number(outputs[outputs.length - 1].activation) || 0;
        return;
      }
      const roots = rootIdsForExport().map(id => nodeById(id)).filter(Boolean);
      const last = roots[roots.length - 1];
      current.result = last ? (Number(last.activation) || 0) : 0;
    }

    function evaluateEditedGraph(renderNow = true) {
      resetBrowserState();
      scanInputSiblings(false);
      propagateDashed(false);
      scanInputSiblings(false);
      averageChildren(false);
      updateRootResultFromEditedGraph();
      document.getElementById("simNote").textContent = "Current browser graph evaluated from the edited structure.";
      if (renderNow) render();
    }

    function updateSelectedNode(field, value) {
      if (selectedKind !== "node") return;
      const node = nodeById(selected);
      if (!node) return;
      if (field === "id") {
        const next = String(value).trim();
        if (!next || (next !== node.id && nodeById(next))) return;
        current.edges.forEach(e => {
          if (e.source === node.id) e.source = next;
          if (e.target === node.id) e.target = next;
        });
        if (positions.has(node.id)) {
          positions.set(next, positions.get(node.id));
          positions.delete(node.id);
        }
        if (current.roots) current.roots = current.roots.map(id => id === node.id ? next : id);
        node.id = next;
        setNodeSelection([next]);
      } else if (field === "icon") {
        node.icon = String(value || "auto");
        if (node.icon !== "custom") delete node.iconImage;
        applyNodeIconEverywhere(node.id, node.icon, node.iconImage || null);
      } else {
        if (field === "gate" && node.type === "return") {
          node.gate = 1.0;
          render();
          return;
        }
        node[field] = Math.max(0, Math.min(1, Number(value)));
      }
      render();
    }

    function applyNodeIconEverywhere(nodeId, icon, iconImage = null) {
      const apply = graphState => {
        const target = graphState?.nodes?.find(n => n.id === nodeId);
        if (!target) return;
        target.icon = icon;
        if (iconImage) target.iconImage = iconImage;
        else delete target.iconImage;
      };
      apply(payload.base);
      payload.frames.forEach(apply);
    }

    function setSelectedIconImage(dataUrl) {
      if (selectedKind !== "node") return;
      const node = nodeById(selected);
      if (!node) return;
      node.icon = "custom";
      node.iconImage = dataUrl;
      applyNodeIconEverywhere(node.id, "custom", dataUrl);
      render();
    }

    function clearSelectedIconImage() {
      if (selectedKind !== "node") return;
      const node = nodeById(selected);
      if (!node) return;
      node.icon = "auto";
      delete node.iconImage;
      applyNodeIconEverywhere(node.id, "auto", null);
      render();
    }

    function classForNodeType(type) {
      return {
        input: "InputNode",
        return: "OutputNode",
        object: "ObjectNode",
        parallel: "ParallelNode",
        serial: "SerialNode",
        body: "BodyNode",
        plan: "PlanNode",
        template: "TemplateNode",
        field: "FieldNode",
        out: "WindowNode",
        in: "InWindowNode",
      }[type] || "ObjectNode";
    }

    function defaultNodeFields(id, type) {
      const base = {
        id,
        type,
        class: classForNodeType(type),
        activation: type === "input" ? 1 : 0,
        gate: 1,
        children: [],
        icon: "auto",
      };
      if (type === "input") {
        return { ...base, modality: "retina", field: null };
      }
      if (type === "return") {
        return {
          ...base,
          role: "output",
          field: { shape: [9, 9], mean: 0, min: 0, max: 0, values: matrixFilled(9, 9, 0) },
        };
      }
      if (type === "object") {
        return {
          ...base,
          pattern: [1, 1, 1],
          pattern_name: "universal",
          name: "universal",
          orientation: "vertical",
          dynamic: false,
        };
      }
      if (type === "serial") {
        return { ...base, split_orientation: "vertical", orientation: "vertical" };
      }
      return base;
    }

    function convertNodeType(node, nextType) {
      const id = node.id;
      const keep = {
        activation: node.activation,
        gate: nextType === "return" ? 1 : node.gate,
        children: Array.isArray(node.children) ? [...node.children] : [],
        icon: node.icon || "auto",
        iconImage: node.iconImage,
      };
      const converted = { ...defaultNodeFields(id, nextType), ...keep };
      if (nextType === "input" && node.field) converted.field = structuredClone(node.field);
      if (nextType === "return" && node.field) converted.field = structuredClone(node.field);
      if (nextType === "object") {
        converted.pattern = Array.isArray(node.pattern) ? [...node.pattern] : converted.pattern;
        converted.pattern_name = node.pattern_name || node.name || converted.pattern_name;
        converted.name = converted.pattern_name;
        converted.orientation = node.orientation || node.split_orientation || converted.orientation;
        converted.dynamic = Boolean(node.dynamic);
      }
      if (nextType === "serial") {
        converted.split_orientation = node.split_orientation || node.orientation || "vertical";
        converted.orientation = converted.split_orientation;
      }
      Object.keys(node).forEach(key => delete node[key]);
      Object.assign(node, converted);
    }

    function changeSelectedNodeType(nextType) {
      if (selectedKind !== "node") return;
      const node = nodeById(selected);
      if (!node || node.type === nextType) return;
      convertNodeType(node, nextType);
      const apply = graphState => {
        const target = graphState?.nodes?.find(n => n.id === node.id);
        if (target) convertNodeType(target, nextType);
      };
      apply(payload.base);
      payload.frames.forEach(apply);
      refreshReturnWindows(false);
      render();
    }

    function addNode() {
      let i = current.nodes.length + 1;
      let id = `node_${i}`;
      while (nodeById(id)) id = `node_${++i}`;
      const action = document.getElementById("addAction")?.value || "node:object";
      const type = action.startsWith("node:") ? action.slice(5) : "object";
      const n = defaultNodeFields(id, type);
      current.nodes.push(n);
      const { w, h } = graphSize();
      positions.set(id, { x: (w / 2 - viewOffset.x) / viewScale, y: (h / 2 - viewOffset.y) / viewScale, vx: 0, vy: 0, level: 1 });
      setNodeSelection([id]);
      render();
    }

    function deleteSelected() {
      if (selectedKind === "node" || selectedKind === "nodes") {
        const ids = selectedKind === "nodes" ? new Set(selectedNodes) : new Set([selected]);
        current.nodes = current.nodes.filter(n => !ids.has(n.id));
        current.edges = current.edges.filter(e => !ids.has(e.source) && !ids.has(e.target));
        if (current.roots) current.roots = current.roots.filter(id => !ids.has(id));
        current.nodes.forEach(n => {
          if (Array.isArray(n.children)) n.children = n.children.filter(id => !ids.has(id));
        });
        ids.forEach(id => positions.delete(id));
      } else if (selectedKind === "edge") {
        const edge = current.edges[selected];
        current.edges.splice(selected, 1);
        if (edge?.type === "solid") {
          const parent = nodeById(edge.source);
          if (parent && Array.isArray(parent.children)) {
            parent.children = parent.children.filter(id => id !== edge.target);
          }
        }
      }
      selected = null;
      selectedKind = null;
      selectedNodes.clear();
      render();
    }

    function edgeExists(source, target, type) {
      return current.edges.some(edge =>
        edge.source === source && edge.target === target && edge.type === type
      );
    }

    function setStructureNote(text) {
      const note = document.getElementById("structureNote");
      if (note) note.textContent = text;
    }

    function addEdge(type, sourceArg = null, targetArg = null) {
      const source = sourceArg || document.getElementById("edgeSource")?.value;
      const target = targetArg || document.getElementById("edgeTarget")?.value;
      if (!source || !target || source === target) return;
      if (edgeExists(source, target, type)) {
        selected = current.edges.findIndex(edge =>
          edge.source === source && edge.target === target && edge.type === type
        );
        selectedKind = "edge";
        selectedNodes.clear();
        setStructureNote(`${type} edge already exists.`);
        render();
        return;
      }
      current.edges.push({ source, target, type });
      if (type === "solid") {
        const parent = nodeById(source);
        if (parent && !parent.children.includes(target)) parent.children.push(target);
      }
      setStructureNote(`${type} edge added.`);
      render();
    }

    function dashedTopologicalOrder() {
      const ids = current.nodes.map(node => node.id);
      const inDegree = new Map(ids.map(id => [id, 0]));
      const successors = new Map(ids.map(id => [id, []]));
      current.edges.filter(edge => edge.type === "dashed").forEach(edge => {
        if (!inDegree.has(edge.source) || !inDegree.has(edge.target)) return;
        inDegree.set(edge.target, (inDegree.get(edge.target) || 0) + 1);
        successors.get(edge.source).push(edge.target);
      });
      const queue = ids.filter(id => (inDegree.get(id) || 0) === 0);
      const order = [];
      while (queue.length) {
        const id = queue.shift();
        order.push(id);
        (successors.get(id) || []).forEach(target => {
          inDegree.set(target, (inDegree.get(target) || 0) - 1);
          if ((inDegree.get(target) || 0) === 0) queue.push(target);
        });
      }
      ids.forEach(id => {
        if (!order.includes(id)) order.push(id);
      });
      return order;
    }

    function propagateDashed(renderNow = true) {
      const incoming = new Map();
      current.edges.filter(e => e.type === "dashed").forEach(e => {
        if (!incoming.has(e.target)) incoming.set(e.target, []);
        const src = nodeById(e.source);
        if (src) incoming.get(e.target).push(Number(src.activation) || 0);
      });
      dashedTopologicalOrder().forEach(target => {
        const values = incoming.get(target);
        if (!values || values.length === 0) return;
        const node = nodeById(target);
        if (!node) return;
        const gate = values.reduce((a, b) => a + b, 0) / values.length;
        node.gate = gate;
        if (node.type === "input" || (node.type === "object" && !fieldForObservedNode(node))) {
          node.activation = gate;
        }
      });
      refreshReturnWindows(false);
      if (renderNow) render();
    }

    function averageChildren(renderNow = true) {
      const solid = current.edges.filter(e => e.type === "solid");
      for (let pass = 0; pass < current.nodes.length; pass++) {
        current.nodes.forEach(n => {
          const children = solid.filter(e => e.source === n.id).map(e => nodeById(e.target)).filter(Boolean);
          if (!children.length) return;
          n.activation = children.reduce((sum, child) => sum + (Number(child.activation) || 0), 0) / children.length;
          if (n.type === "return") n.gate = 1.0;
        });
      }
      refreshReturnWindows(false);
      if (renderNow) render();
    }

    function play() {
      if (timer) {
        clearInterval(timer);
        timer = null;
        document.getElementById("playBtn").textContent = "Play";
        return;
      }
      document.getElementById("playBtn").textContent = "Pause";
      timer = setInterval(() => {
        if (frameIndex >= payload.frames.length - 1) {
          applyFrame(0);
        } else {
          applyFrame(frameIndex + 1);
        }
      }, 650);
    }

    function solidEdges() {
      return current.edges.filter(edge => edge.type === "solid");
    }

    function dashedEdges() {
      return current.edges.filter(edge => edge.type === "dashed");
    }

    function solidChildrenByParent() {
      const children = new Map(current.nodes.map(node => [node.id, []]));
      solidEdges().forEach(edge => {
        if (children.has(edge.source) && nodeById(edge.target)) children.get(edge.source).push(edge.target);
      });
      return children;
    }

    function rootIdsForExport() {
      if (Array.isArray(current.roots) && current.roots.length) {
        return current.roots.filter(id => nodeById(id));
      }
      const targets = new Set(solidEdges().map(edge => edge.target));
      return current.nodes.map(node => node.id).filter(id => !targets.has(id));
    }

    function nodeLabel(node) {
      if (!node) return "UnknownNode(?)";
      if (node.type === "input") return `InputNode("${node.id}")`;
      if (node.type === "return") return `OutputNode("${node.id}")`;
      if (node.type === "object") {
        const name = node.pattern_name || node.name || "object";
        return `ObjectNode("${node.id}", name="${name}")`;
      }
      const cls = {
        parallel: "ParallelNode",
        serial: "SerialNode",
        body: "BodyNode",
        plan: "PlanNode",
        template: "TemplateNode",
        field: "FieldNode",
        out: "WindowNode",
        in: "InWindowNode",
      }[node.type] || (node.class || "Node");
      return `${cls}("${node.id}")`;
    }

    function exportTree() {
      const children = solidChildrenByParent();
      const roots = rootIdsForExport();
      const lines = [];
      const seen = new Set();
      function walk(id, depth) {
        const node = nodeById(id);
        if (!node) return;
        lines.push(`${"  ".repeat(depth)}${nodeLabel(node)}`);
        if (seen.has(id)) {
          lines.push(`${"  ".repeat(depth + 1)}...`);
          return;
        }
        seen.add(id);
        (children.get(id) || []).forEach(childId => walk(childId, depth + 1));
      }
      roots.forEach((rootId, index) => {
        if (index) lines.push("");
        walk(rootId, 0);
      });
      const dashed = dashedEdges();
      if (dashed.length) {
        lines.push("", "# dashed gates");
        dashed.forEach(edge => lines.push(`${edge.source} --dashed--> ${edge.target}`));
      }
      return lines.join("\n");
    }

    function pyName(id) {
      let name = String(id).replace(/[^A-Za-z0-9_]/g, "_");
      if (!/^[A-Za-z_]/.test(name)) name = `n_${name}`;
      const reserved = new Set(["I", "O", "in", "for", "if", "else", "class", "def", "return"]);
      if (reserved.has(name)) name = `${name}_node`;
      return name || "node";
    }

    function pyStr(value) {
      return JSON.stringify(String(value));
    }

    function pyPattern(node) {
      if (!Array.isArray(node.pattern)) return 'TRIGRAM_PATTERNS["neutral"]';
      const match = trigramPalette.find(item =>
        item.pattern.length === node.pattern.length &&
        item.pattern.every((v, i) => Number(v) === Number(node.pattern[i]))
      );
      return match ? `TRIGRAM_PATTERNS[${pyStr(match.name)}]` : `[${node.pattern.map(v => Number(v)).join(", ")}]`;
    }

    function pyOrientation(node) {
      return (node.orientation || node.split_orientation || "vertical") === "horizontal"
        ? "Orientation.HORIZONTAL"
        : "Orientation.VERTICAL";
    }

    function pyConstructor(node) {
      const id = pyStr(node.id);
      if (node.type === "input") return `InputNode(${id}, modality=${pyStr(node.modality || "retina")})`;
      if (node.type === "return") return `OutputNode(${id})`;
      if (node.type === "object") {
        const args = [id, pyPattern(node), pyOrientation(node)];
        if (node.pattern_name || node.name) args.push(`name=${pyStr(node.pattern_name || node.name)}`);
        if (node.dynamic) args.push("is_dynamic=True");
        return `ObjectNode(${args.join(", ")})`;
      }
      if (node.type === "serial") return `SerialNode(${id}, split_orientation=${pyOrientation(node)})`;
      if (node.type === "body") return `BodyNode(${id})`;
      if (node.type === "plan") return `PlanNode(${id})`;
      if (node.type === "template") return `TemplateNode(${id})`;
      if (node.type === "field") return `FieldNode(${id})`;
      if (node.type === "out") return `WindowNode(${id})`;
      if (node.type === "in") return `InWindowNode(${id})`;
      return `ParallelNode(${id})`;
    }

    function exportPython() {
      const names = new Map();
      const used = new Set();
      current.nodes.forEach(node => {
        let base = pyName(node.id);
        let name = base;
        let i = 2;
        while (used.has(name)) name = `${base}_${i++}`;
        used.add(name);
        names.set(node.id, name);
      });
      const lines = [
        "from embodied_lot import ConceptGraph, InputNode, OutputNode, ObjectNode, ParallelNode, SerialNode, BodyNode, PlanNode, TemplateNode, FieldNode, WindowNode, InWindowNode, Orientation, TRIGRAM_PATTERNS",
        "",
        `g = ConceptGraph(${pyStr(current.graph_name || payload.name || "edited_graph")})`,
      ];
      current.nodes.forEach(node => {
        lines.push(`${names.get(node.id)} = ${pyConstructor(node)}`);
      });
      if (current.nodes.length) {
        lines.push(`g.add_nodes([${current.nodes.map(node => names.get(node.id)).join(", ")}])`);
      }
      solidEdges().forEach(edge => {
        if (names.has(edge.source) && names.has(edge.target)) {
          lines.push(`g.add_solid_edge(${names.get(edge.source)}, ${names.get(edge.target)})`);
        }
      });
      dashedEdges().forEach(edge => {
        if (names.has(edge.source) && names.has(edge.target)) {
          lines.push(`g.add_dashed_edge(${names.get(edge.source)}, ${names.get(edge.target)})`);
        }
      });
      const roots = rootIdsForExport();
      if (roots.length === 1 && names.has(roots[0])) {
        lines.push(`g.root = ${names.get(roots[0])}`);
      } else if (roots.length > 1) {
        lines.push(`# roots: ${roots.map(id => names.get(id) || id).join(", ")}`);
      }
      return lines.join("\n");
    }

    function exportJson() {
      return JSON.stringify({
        graph_name: current.graph_name || payload.name,
        roots: rootIdsForExport(),
        nodes: current.nodes,
        edges: current.edges,
      }, null, 2);
    }

    function structureExportText() {
      const format = document.getElementById("exportFormat")?.value || "tree";
      if (format === "program") {
        try { return graphToProgram(); } catch (error) { return `# ${error.message}`; }
      }
      if (format === "python") return exportPython();
      if (format === "json") return exportJson();
      return exportTree();
    }

    function renderStructureExport() {
      const out = document.getElementById("structureExport");
      if (!out) return;
      out.textContent = structureExportText();
    }

    async function copyStructureExport() {
      const text = structureExportText();
      try {
        await navigator.clipboard.writeText(text);
      } catch (error) {
        const textarea = document.createElement("textarea");
        textarea.value = text;
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand("copy");
        textarea.remove();
      }
    }

    function escapeHtml(value) {
      return String(value).replace(/[&<>"']/g, ch => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" }[ch]));
    }

    function renderLegend() {
      const legend = document.getElementById("legend");
      legend.innerHTML = nodeTypes
        .map(name => `<span><i class="swatch" style="background:${nodeColour(name)};width:8px;height:8px;border-radius:50%;display:inline-block;margin-right:5px;vertical-align:-1px"></i>${name === "return" ? "output" : name}</span>`)
        .join("");
    }

    function updateIconSizeControl(value) {
      const slider = document.getElementById("iconSizeSlider");
      const label = document.getElementById("iconSizeValue");
      const percent = Math.max(70, Math.min(135, Number(value ?? slider?.value ?? 95)));
      labIconScale = percent / 100;
      if (slider) slider.value = String(percent);
      if (label) label.textContent = `${percent}%`;
      render();
    }

    document.getElementById("prevBtn").addEventListener("click", () => applyFrame(frameIndex - 1));
    document.getElementById("nextBtn").addEventListener("click", () => applyFrame(frameIndex + 1));
    document.getElementById("playBtn").addEventListener("click", play);
    document.getElementById("layoutBtn").addEventListener("click", resetLayoutOnce);
    document.getElementById("solidToggle").addEventListener("change", render);
    document.getElementById("dashedToggle").addEventListener("change", render);
    document.getElementById("freeToggle").addEventListener("change", () => applyFrame(frameIndex));
    document.getElementById("labviewToggle").addEventListener("change", e => {
      document.body.classList.toggle("labview", e.target.checked);
      render();
    });
    document.getElementById("iconSizeSlider").addEventListener("input", e => updateIconSizeControl(e.target.value));
    document.getElementById("floatToggle").addEventListener("change", e => {
      if (e.target.checked) startPhysics();
      else stopPhysics();
    });
    timeline.addEventListener("input", e => applyFrame(Number(e.target.value)));
    svg.addEventListener("pointerdown", marqueeStart);
    svg.addEventListener("pointermove", updatePendingPreview);
    svg.addEventListener("wheel", zoomGraph, { passive: false });
    svg.addEventListener("click", evt => {
      if (evt.target !== svg) return;
      if (suppressNextCanvasClick) {
        suppressNextCanvasClick = false;
        return;
      }
      if (editMode === "node") {
        const p = graphPoint(evt);
        addNode();
        const id = selected;
        if (id && positions.has(id)) {
          positions.set(id, { ...positions.get(id), x: p.x, y: p.y });
          render();
        }
        return;
      }
      pendingEdge = null;
      previewPoint = null;
      setNodeSelection([]);
      render();
    });
    document.getElementById("nodeId").addEventListener("change", e => updateSelectedNode("id", e.target.value));
    document.getElementById("nodeType").addEventListener("change", e => changeSelectedNodeType(e.target.value));
    document.getElementById("nodeIcon").addEventListener("change", e => updateSelectedNode("icon", e.target.value));
    document.getElementById("nodeIconFile").addEventListener("change", e => {
      const file = e.target.files?.[0];
      if (!file) return;
      if (!file.type.startsWith("image/")) {
        e.target.value = "";
        return;
      }
      const reader = new FileReader();
      reader.addEventListener("load", () => setSelectedIconImage(String(reader.result)));
      reader.readAsDataURL(file);
    });
    document.getElementById("clearIconSpriteBtn").addEventListener("click", clearSelectedIconImage);
    document.getElementById("nodeGate").addEventListener("change", e => updateSelectedNode("gate", e.target.value));
    document.getElementById("nodeActivation").addEventListener("change", e => updateSelectedNode("activation", e.target.value));
    document.getElementById("applyFieldBtn").addEventListener("click", () => {
      try {
        setSelectedField(parseFieldText(document.getElementById("fieldEditor").value));
      } catch (error) {
        document.getElementById("fieldNote").textContent = error.message;
      }
    });
    document.getElementById("applyPatternBtn").addEventListener("click", applySelectedPattern);
    document.getElementById("modeSelectBtn").addEventListener("click", () => setEditMode("select"));
    document.getElementById("addBtn").addEventListener("click", () => {
      const action = document.getElementById("addAction").value;
      if (action === "solid" || action === "dashed") {
        setEditMode(action);
      } else {
        setEditMode("node");
      }
    });
    document.getElementById("removeBtn").addEventListener("click", deleteSelected);
    document.getElementById("scanBtn").addEventListener("click", scanInputSiblings);
    document.getElementById("propagateBtn").addEventListener("click", propagateDashed);
    document.getElementById("averageBtn").addEventListener("click", averageChildren);
    document.getElementById("exportFormat").addEventListener("change", renderStructureExport);
    document.getElementById("copyExportBtn").addEventListener("click", copyStructureExport);
    window.addEventListener("resize", () => { layout(); render(); });
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { renderLegend(); render(); });


    // ── v7 program text and the real engine ───────────────────────────
    // Program text is the search's own syntax: trees side by side, then "| .a.b>.c.d" dashed edges
    // between tree paths. Compiled programs wrap each tree as par_O<k>[O<k>, tree]; those wrappers
    // (and legacy input/output nodes) are skipped when writing a graph back as text.

    const LEAF_NAMES = ["void", "bottom", "center", "lower", "top", "periphery", "upper", "universal", "neutral"];

    function isWrapper(node) {
      return node && node.type === "parallel" && /^par_O\d*$/.test(node.id);
    }

    function programTrees() {
      const kids = solidChildrenByParent();
      const trees = [];
      rootIdsForExport().forEach(id => {
        const node = nodeById(id);
        if (!node || node.type === "input" || node.type === "return") return;
        if (isWrapper(node)) {
          (kids.get(id) || []).map(nodeById).filter(n => n && n.type !== "return" && n.type !== "input")
            .forEach(n => trees.push(n.id));
        } else {
          trees.push(id);
        }
      });
      return trees;
    }

    function leafName(node) {
      const name = node.pattern_name || node.name;
      if (LEAF_NAMES.includes(name)) return name;
      const match = trigramPalette.find(item => Array.isArray(node.pattern) &&
        item.pattern.length === node.pattern.length && item.pattern.every((v, i) => Number(v) === Number(node.pattern[i])));
      if (!match) throw new Error(`${node.id}: pattern [${node.pattern}] is not a trigram`);
      return match.name;
    }

    function graphToProgram() {
      const kids = solidChildrenByParent();
      const paths = new Map();
      const trees = programTrees();
      if (!trees.length) throw new Error("the graph has no tree to write");
      const forest = trees.length > 1;
      const seen = new Set();
      function write(id, path) {
        const node = nodeById(id);
        if (seen.has(id)) throw new Error(`${id} has two parents: a program is a tree`);
        seen.add(id);
        paths.set(id, path);
        const children = (kids.get(id) || []).filter(c => {
          const t = nodeById(c)?.type;
          return t !== "input" && t !== "return";
        });
        const sub = () => children.map((c, i) => write(c, [...path, i])).join(" ");
        if (node.type === "object") {
          const ori = (node.orientation || "vertical") === "horizontal" ? "H" : "V";
          return `${leafName(node)}/${ori}${node.dynamic ? "*" : ""}`;
        }
        if (node.type === "out" || node.type === "in") return node.type;
        if (!children.length) throw new Error(`${id} (${node.type}) has no children`);
        if (node.type === "serial") {
          return `(serial_${(node.split_orientation || "vertical") === "horizontal" ? "h" : "v"} ${sub()})`;
        }
        if (["parallel", "template", "field", "body", "plan"].includes(node.type)) {
          return `(${node.type === "plan" ? "template" : node.type} ${sub()})`;
        }
        throw new Error(`${id}: ${node.type} nodes have no program syntax`);
      }
      const text = trees.map((id, k) => write(id, forest ? [k] : [])).join(" ");
      const edges = dashedEdges().map(e => {
        if (!paths.has(e.source) || !paths.has(e.target)) throw new Error(`dashed edge ${e.source} → ${e.target} leaves the program`);
        return `.${paths.get(e.source).join(".")}>.${paths.get(e.target).join(".")}`;
      });
      return edges.length ? `${text} | ${edges.join(" ")}` : text;
    }

    let engineRun = null;
    let engineStates = {};

    function engineAvailable() {
      return Boolean(payload.engine) && location.protocol.startsWith("http");
    }

    function engineMessage(html) {
      document.getElementById("engineMsgs").innerHTML = html;
    }

    async function enginePost(path, body) {
      const response = await fetch(path, {method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify(body)});
      const data = await response.json();
      if (data.error) throw Object.assign(new Error(data.error), {trace: data.trace});
      return data;
    }

    function showEngineError(error) {
      engineMessage(`<div class="bug"><b>${escapeHtml(error.message)}</b>${error.trace ? "\n" + escapeHtml(error.trace) : ""}</div>`);
    }

    function issuesHtml(issues) {
      return issues.length
        ? `<div class="warn"><b>Outside the configured language</b> (the engine still runs it):\n${issues.map(i => "· " + escapeHtml(i)).join("\n")}</div>`
        : "";
    }

    function replaceGraph(base, frames) {
      payload.base = base;
      payload.frames = frames;
      positions = new Map();
      selected = null; selectedKind = null; selectedNodes.clear();
      current = structuredClone(frames[0] || base);
      layout();
      applyFrame(0);
    }

    async function loadProgram() {
      try {
        const data = await enginePost("/api/graph", {program: document.getElementById("programText").value});
        engineRun = null;
        engineStates = {};
        document.getElementById("engineUnits").innerHTML = "";
        payload.name = data.program;
        replaceGraph(data.base, data.frames);
        document.getElementById("programText").value = data.program;
        engineMessage(issuesHtml(data.issues) || `<div class="ok">Loaded. Edit the graph, then Graph → Program, or Run on Engine.</div>`);
      } catch (error) { showEngineError(error); }
    }

    async function runEngine() {
      const button = document.getElementById("runEngineBtn");
      button.disabled = true; button.textContent = "Running…";
      try {
        const data = await enginePost("/api/run", {program: document.getElementById("programText").value,
          ticks: Number(document.getElementById("engineTicks").value) || 24, states: engineStates});
        engineRun = data;
        engineStates = Object.fromEntries(data.units.map(u => [u.key, u.state]));
        document.getElementById("programText").value = data.program;
        renderEngineUnits();
        const msgs = [];
        if (data.bugs.length) msgs.push(`<div class="bug"><b>Engine inconsistency</b>\n${data.bugs.map(b => "· " + escapeHtml(b)).join("\n")}</div>`);
        msgs.push(issuesHtml(data.issues));
        msgs.push(`<div class="ok">${data.snapshots.length} ticks · ${data.windows.length} output window${data.windows.length === 1 ? "" : "s"} · ${data.combinations} motion realization${data.combinations === 1 ? "" : "s"}${data.bugs.length ? "" : " · recorder and component rollouts agree with the engine"}. Replay with Play or the timeline.</div>`);
        engineMessage(msgs.join(""));
        payload.name = data.program;
        replaceGraph(data.snapshots[0], data.snapshots);
      } catch (error) {
        showEngineError(error);
      } finally {
        button.disabled = false; button.textContent = "Run on Engine";
      }
    }

    function renderEngineUnits() {
      const box = document.getElementById("engineUnits");
      if (!engineRun || !engineRun.units.length) { box.innerHTML = ""; return; }
      box.innerHTML = "unknown motion:" + engineRun.units.map(u =>
        `<label>${escapeHtml(u.key)} <select data-key="${escapeHtml(u.key)}">${u.options.map(o =>
          `<option${o === u.state ? " selected" : ""}>${o}</option>`).join("")}</select></label>`).join("");
      box.querySelectorAll("select").forEach(sel => sel.addEventListener("change", () => {
        engineStates[sel.dataset.key] = sel.value;
        runEngine();
      }));
    }

    function boardFigure(caption, values) {
      const fig = document.createElement("figure");
      const cap = document.createElement("figcaption");
      cap.textContent = caption;
      const grid = document.createElement("div");
      grid.className = "pixel-preview";
      grid.style.gridTemplateRows = `repeat(${values.length}, 1fr)`;
      grid.style.gridTemplateColumns = `repeat(${values[0].length}, 1fr)`;
      values.forEach((row, r) => row.forEach((value, c) => {
        const cell = document.createElement("span");
        cell.style.background = fieldColour(Math.max(0, Math.min(1, Number(value))));
        cell.title = `row ${r}, column ${c}: ${Number(value).toFixed(2)}`;
        grid.appendChild(cell);
      }));
      fig.append(cap, grid);
      return fig;
    }

    function renderEngineBoards() {
      const box = document.getElementById("engineBoards");
      if (!box) return;
      box.innerHTML = "";
      const f = engineRun && current.engine ? engineRun.frames[frameIndex] : null;
      if (!f) return;
      box.appendChild(boardFigure(`drawn, tick ${f.t}`, f.drawn));
      if (engineRun.roots.length > 1) engineRun.roots.forEach((r, i) => box.appendChild(boardFigure(`tree ${i} space`, f.spaces[r])));
      (engineRun.channels[frameIndex + 1] || []).forEach((img, k) =>
        box.appendChild(boardFigure(`out ${k} (${engineRun.windows[k] || "tree"})`, img)));
    }

    function initEnginePanel() {
      const select = document.getElementById("engineExamples");
      if (!engineAvailable()) {
        document.querySelectorAll("#enginePanel select, #enginePanel textarea, #enginePanel input, #enginePanel button:not(#graphToProgramBtn)")
          .forEach(el => { el.disabled = true; });
        engineMessage(`<div class="warn">No engine here (the page was opened as a file). To run programs on the engine, start the server:\npython -m grace_lot.analysis.playground\nand open http://localhost:8765/graph. Graph → Program still works here.</div>`);
        return;
      }
      payload.engine.examples.forEach((e, i) => {
        const o = document.createElement("option");
        o.value = i; o.textContent = e.title;
        select.appendChild(o);
      });
      select.addEventListener("change", () => {
        document.getElementById("programText").value = payload.engine.examples[Number(select.value)].program;
        engineStates = {};
        loadProgram();
      });
      document.getElementById("programText").value = payload.program || payload.engine.examples[0].program;
      engineMessage(issuesHtml(payload.issues || []) || `<div class="ok">Semantics from ${escapeHtml(payload.engine.config)}.</div>`);
      document.getElementById("loadProgramBtn").addEventListener("click", loadProgram);
      document.getElementById("runEngineBtn").addEventListener("click", runEngine);
      document.getElementById("programText").addEventListener("keydown", e => {
        if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); runEngine(); }
      });
    }

    document.getElementById("graphToProgramBtn").addEventListener("click", () => {
      try {
        document.getElementById("programText").value = graphToProgram();
        engineMessage(`<div class="ok">Program written from the graph.</div>`);
      } catch (error) { showEngineError(error); }
    });

    document.getElementById("iconSizeValue").textContent = `${Math.round(labIconScale * 100)}%`;
    renderLegend();
    renderTrigramPalette();
    populateIconSelect();
    initEnginePanel();
    layout();
    render();
