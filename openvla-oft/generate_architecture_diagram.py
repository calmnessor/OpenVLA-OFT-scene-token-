"""
Generate OpenVLA-OFT Model Architecture Diagram in Draw.io (.drawio) format.
Feature dimension transformations are clearly annotated at each step.
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom

# ==============================================================================
# Draw.io XML constants
# ==============================================================================
DRAWIO_NS = "http://www.w3.org/1999/xhtml"

BOX_STYLE = (
    "rounded=1;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;"
    "fontFamily=Consolas;fontSize=11;align=center;"
)
BOX_STYLE_VISION = (
    "rounded=1;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;"
    "fontFamily=Consolas;fontSize=11;align=center;"
)
BOX_STYLE_PROJ = (
    "rounded=1;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;"
    "fontFamily=Consolas;fontSize=11;align=center;"
)
BOX_STYLE_LLM = (
    "rounded=1;whiteSpace=wrap;html=1;fillColor=#f8cecc;strokeColor=#b85450;"
    "fontFamily=Consolas;fontSize=11;align=center;"
)
BOX_STYLE_HEAD = (
    "rounded=1;whiteSpace=wrap;html=1;fillColor=#e1d5e7;strokeColor=#9673a6;"
    "fontFamily=Consolas;fontSize=11;align=center;"
)
BOX_STYLE_INPUT = (
    "rounded=1;whiteSpace=wrap;html=1;fillColor=#f5f5f5;strokeColor=#666666;"
    "fontFamily=Consolas;fontSize=11;align=center;"
)
BOX_STYLE_AUX = (
    "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffe6cc;strokeColor=#d79b00;"
    "fontFamily=Consolas;fontSize=11;align=center;"
)
BOX_STYLE_SECTION = (
    "rounded=0;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#999999;"
    "fontFamily=Consolas;fontSize=10;align=left;verticalAlign=top;dashed=1;"
)
EDGE_STYLE = (
    "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;"
    "html=1;strokeColor=#666666;fontFamily=Consolas;fontSize=9;"
)

BOX_W, BOX_H = 170, 52
SMALL_W, SMALL_H = 150, 36
WIDE_W, WIDE_H = 210, 52
TALL_W, TALL_H = 170, 90
LLM_W, LLM_H = 180, 130

# ==============================================================================
# Helper functions
# ==============================================================================
def create_mxfile():
    root = ET.Element("mxfile")
    root.set("host", "app.diagrams.net")
    root.set("modified", "2025-01-01T00:00:00.000Z")
    root.set("agent", "generated")
    root.set("version", "24.0.0")
    root.set("type", "device")

    diagram = ET.SubElement(root, "diagram")
    diagram.set("id", "openvla-oft-arch")
    diagram.set("name", "OpenVLA-OFT Architecture")

    mxGraphModel = ET.SubElement(diagram, "mxGraphModel")
    mxGraphModel.set("dx", "1422")
    mxGraphModel.set("dy", "850")
    mxGraphModel.set("grid", "1")
    mxGraphModel.set("gridSize", "10")
    mxGraphModel.set("guides", "1")
    mxGraphModel.set("tooltips", "1")
    mxGraphModel.set("connect", "1")
    mxGraphModel.set("arrows", "1")
    mxGraphModel.set("fold", "1")
    mxGraphModel.set("page", "1")
    mxGraphModel.set("pageScale", "1")
    mxGraphModel.set("pageWidth", "2000")
    mxGraphModel.set("pageHeight", "1200")
    mxGraphModel.set("math", "0")
    mxGraphModel.set("shadow", "0")

    root_elem = ET.SubElement(mxGraphModel, "root")

    # Default cell
    ET.SubElement(root_elem, "mxCell", id="0")
    ET.SubElement(root_elem, "mxCell", id="1", parent="0")

    return root, root_elem

def add_cell(root_elem, cell_id, value, style, x, y, w, h, parent="1", vertex="1"):
    cell = ET.SubElement(root_elem, "mxCell")
    cell.set("id", cell_id)
    cell.set("value", value)
    cell.set("style", style)
    cell.set("vertex", vertex)
    cell.set("parent", parent)
    geom = ET.SubElement(cell, "mxGeometry")
    geom.set("x", str(x))
    geom.set("y", str(y))
    geom.set("width", str(w))
    geom.set("height", str(h))
    geom.set("as", "geometry")
    return cell

def add_edge(root_elem, edge_id, source_id, target_id, value="", style=EDGE_STYLE):
    cell = ET.SubElement(root_elem, "mxCell")
    cell.set("id", edge_id)
    cell.set("value", value)
    cell.set("style", style)
    cell.set("edge", "1")
    cell.set("parent", "1")
    cell.set("source", source_id)
    cell.set("target", target_id)
    geom = ET.SubElement(cell, "mxGeometry")
    geom.set("relative", "1")
    geom.set("as", "geometry")
    return cell

# ==============================================================================
# Build Diagram
# ==============================================================================
root, r = create_mxfile()

# ---- Layout coordinates ----
# We use a column-based layout:
# Col 0 (x=80):  Input images / Text / Proprio labels
# Col 1 (x=280): Image Preprocessing
# Col 2 (x=520): Vision Backbones (SigLIP + DINOv2)
# Col 3 (x=800): Projector → LLM space
# Col 4 (x=1040): Multi-modal Assembly
# Col 5 (x=1300): LLM Backbone (Llama-2 7B)
# Col 6 (x=1560): Action Head → Output

# y positions
Y_IMG_TOP = 60      # Top image branch
Y_IMG_MID = 200     # Second image (wrist)
Y_VGGT = 340        # VGGT scene tokens
Y_TEXT = 460        # Text prompt
Y_PROPRIO = 560     # Proprio branch
Y_LLM_CENTER = 260  # LLM center

# ==============================================================================
# SECTION BORDERS
# ==============================================================================
add_cell(r, "sec1", "Image Preprocessing", BOX_STYLE_SECTION, 270, 30, 190, 410)

add_cell(r, "sec2", "Vision Backbones\n(Dual: SigLIP + DINOv2)", BOX_STYLE_SECTION, 500, 30, 240, 410)

add_cell(r, "sec3", "Projection\nto LLM Space", BOX_STYLE_SECTION, 780, 30, 190, 410)

add_cell(r, "sec4", "Multimodal\nAssembly", BOX_STYLE_SECTION, 1020, 30, 210, 570)

add_cell(r, "sec5", "LLM Backbone\n(Llama-2 7B)", BOX_STYLE_SECTION, 1280, 30, 200, 570)

add_cell(r, "sec6", "Action Head\n& Decoding", BOX_STYLE_SECTION, 1540, 30, 200, 570)

# ==============================================================================
# ROW 0: PRIMARY CAMERA IMAGE
# ==============================================================================
add_cell(r, "img1", "Primary Camera\n(3rd-person view)\nnumpy [H, W, 3] uint8", BOX_STYLE_INPUT,
         80, Y_IMG_TOP, BOX_W, BOX_H)

add_cell(r, "prep1", "LetterboxPad\n→ Resize(224)\n→ CenterCrop(224)\n→ ToTensor → Normalize", BOX_STYLE,
         280, Y_IMG_TOP-10, BOX_W, BOX_H+20)

add_cell(r, "pp_out1", "pixel_values\n[3, 224, 224]", BOX_STYLE,
         280, Y_IMG_TOP+70, BOX_W, 26)

add_edge(r, "e1", "img1", "prep1")

# ---- SigLIP branch ----
add_cell(r, "siglip1", "SigLIP ViT-SO400M\npatch_embed 14×14\n→ 256 patches\nOutput: [256, 1152]", BOX_STYLE_VISION,
         520, Y_IMG_TOP-30, BOX_W, BOX_H+10)

# ---- DINOv2 branch ----
add_cell(r, "dino1", "DINOv2 ViT-L/14\npatch_embed 14×14\n→ 256 patches\nOutput: [256, 1024]", BOX_STYLE_VISION,
         520, Y_IMG_TOP+50, BOX_W, BOX_H+10)

add_cell(r, "cat1", "Concat (hidden dim)\n→ [256, 1152+1024]\n= [256, 2176]", BOX_STYLE_VISION,
         520, Y_IMG_TOP+125, SMALL_W, 46)

add_edge(r, "e1s", "pp_out1", "siglip1", "3 ch → SigLIP")
add_edge(r, "e1d", "pp_out1", "dino1", "3 ch → DINOv2")
add_edge(r, "e1sc", "siglip1", "cat1")
add_edge(r, "e1dc", "dino1", "cat1")

# ---- Projector ----
add_cell(r, "proj1", "PrismaticProjector\n(fused backbone)\nfc1: 2176→8704 + GELU\nfc2: 8704→4096 + GELU\nfc3: 4096→4096", BOX_STYLE_PROJ,
         800, Y_IMG_TOP+20, WIDE_W, BOX_H+30)

add_cell(r, "proj_out1", "Vision Embeddings\n[256, 4096]", BOX_STYLE_PROJ,
         800, Y_IMG_TOP+120, WIDE_W, 26)

add_edge(r, "e1p", "cat1", "proj1")
add_edge(r, "e1po", "proj1", "proj_out1")

# ==============================================================================
# ROW 1: WRIST CAMERA IMAGE (optional, num_images_in_input > 1)
# ==============================================================================
add_cell(r, "img2", "Wrist Camera\n(optional)\nnumpy [H, W, 3] uint8", BOX_STYLE_INPUT,
         80, Y_IMG_MID, BOX_W, BOX_H)

add_cell(r, "prep2", "Same preprocessing\nas primary image", BOX_STYLE,
         280, Y_IMG_MID+5, BOX_W, 40)

add_cell(r, "pp_out2", "pixel_values_wrist\n[3, 224, 224]", BOX_STYLE,
         280, Y_IMG_MID+55, BOX_W, 26)

add_edge(r, "e2", "img2", "prep2")

add_cell(r, "siglip2", "SigLIP ViT-SO400M\n→ [256, 1152]", BOX_STYLE_VISION,
         520, Y_IMG_MID-15, BOX_W, 36)
add_cell(r, "dino2", "DINOv2 ViT-L/14\n→ [256, 1024]", BOX_STYLE_VISION,
         520, Y_IMG_MID+30, BOX_W, 36)
add_cell(r, "cat2", "Concat → [256, 2176]", BOX_STYLE_VISION,
         520, Y_IMG_MID+72, BOX_W, 26)

add_edge(r, "e2s", "pp_out2", "siglip2")
add_edge(r, "e2d", "pp_out2", "dino2")
add_edge(r, "e2sc", "siglip2", "cat2")
add_edge(r, "e2dc", "dino2", "cat2")

add_cell(r, "proj2", "PrismaticProjector\n→ [256, 4096]", BOX_STYLE_PROJ,
         800, Y_IMG_MID+15, WIDE_W, 40)

add_edge(r, "e2p", "cat2", "proj2")

# ---- Vision Concat (multi-image) ----
add_cell(r, "vision_concat", "Concat (patch dim)\nPrimary [256, 4096]\n+ Wrist [256, 4096]\n= [512, 4096]",
         BOX_STYLE_VISION, 800, Y_IMG_MID+65, WIDE_W, 56)

add_edge(r, "evc1", "proj_out1", "vision_concat")
add_edge(r, "evc2", "proj2", "vision_concat")

# ==============================================================================
# ROW 2: VGGT-OMEGA SCENE TOKENS (optional)
# ==============================================================================
add_cell(r, "vggt_in", "VGGT-Omega\nScene Token Extractor\n(frozen, no grad)\ninput: [N, 3, 512, 512]", BOX_STYLE_AUX,
         80, Y_VGGT, BOX_W, 70)

add_cell(r, "vggt_out", "Register Tokens\n[N×16, 1024]\n(16 registers/frame)", BOX_STYLE_AUX,
         280, Y_VGGT+10, BOX_W, 50)

add_cell(r, "scene_proj", "SceneProjector\nLinear(1024→2048) + LayerNorm\n→ [N×16, 2048]", BOX_STYLE_AUX,
         520, Y_VGGT+15, BOX_W, 46)

add_edge(r, "evggt1", "vggt_in", "vggt_out")
add_edge(r, "evggt2", "vggt_out", "scene_proj")

# ==============================================================================
# ROW 3: TEXT INPUT
# ==============================================================================
add_cell(r, "text_in", '<font style="font-size:10px">Text Prompt</font>\n<b>"In: What action should\nthe robot take to {task}?\nOut:"</b>', BOX_STYLE_INPUT,
         80, Y_TEXT, BOX_W, BOX_H+10)

add_cell(r, "tokenizer", "Llama-2 Tokenizer\n→ input_ids\n[B, seq_len]", BOX_STYLE,
         280, Y_TEXT+10, BOX_W, 40)

add_cell(r, "tok_emb", "Token Embedding\nnn.Embedding\n→ [B, seq_len, 4096]", BOX_STYLE,
         520, Y_TEXT+10, BOX_W, 40)

add_edge(r, "et1", "text_in", "tokenizer")
add_edge(r, "et2", "tokenizer", "tok_emb")

# ==============================================================================
# ROW 4: PROPRIOCEPTIVE STATE (optional)
# ==============================================================================
add_cell(r, "proprio_in", "Proprioceptive State\n[8] (joints+gripper)", BOX_STYLE_INPUT,
         80, Y_PROPRIO, BOX_W, 46)

add_cell(r, "proprio_proj", "ProprioProjector\nfc1: 8→4096 + GELU\nfc2: 4096→4096\n→ [1, 4096]", BOX_STYLE_AUX,
         280, Y_PROPRIO, BOX_W, BOX_H)

add_edge(r, "ep1", "proprio_in", "proprio_proj")

# ==============================================================================
# MULTIMODAL ASSEMBLY
# ==============================================================================
add_cell(r, "assembly",
         '<b>Sequence Assembly</b>\n'
         '[BOS] patches [text] [actions] [STOP]\n'
         '──────────────────────────\n'
         'BOS: 1 token\n'
         'Vision patches: 256×N\n'
         '+ optional: proprio(1), scene(N×16),\n'
         '  diffusion_timestep(1)\n'
         'Text: seq_len tokens\n'
         'Action placeholder: 56 tokens (zeros)\n'
         '  = 8 chunks × 7 DoF\n'
         'STOP: 1 token\n'
         '──────────────────────────\n'
         'Total: [B, total_seq, 4096]',
         BOX_STYLE, 1040, 80, BOX_W+30, 220)

# Edges to assembly
add_edge(r, "ea_vis", "vision_concat", "assembly")
add_edge(r, "ea_scene", "scene_proj", "assembly")
add_edge(r, "ea_text", "tok_emb", "assembly")
add_edge(r, "ea_proprio", "proprio_proj", "assembly")

# ==============================================================================
# LLM BACKBONE
# ==============================================================================
add_cell(r, "llm",
         '<b>Llama-2 7B</b>\n'
         '──────────────────\n'
         '32 Transformer Layers\n'
         'd_model = 4096\n'
         'num_heads = 32\n'
         'head_dim = 128\n'
         'FFN dim = 11008\n'
         '──────────────────\n'
         '<b>Bidirectional Attention</b>\n'
         '(custom transformers fork)\n'
         '──────────────────\n'
         'Pre-LN architecture\n'
         'RoPE positional encoding\n'
         'Output: [B, total_seq, 4096]',
         BOX_STYLE_LLM, 1300, 80, LLM_W, LLM_H+40)

add_edge(r, "ellm", "assembly", "llm",
         "Multimodal Embeddings\n→ Llama-2 forward()")

# ==============================================================================
# ACTION HEAD
# ==============================================================================
add_cell(r, "extract",
         '<b>Extract Action Hidden States</b>\n'
         'Positions: last 56 tokens\n'
         '(before STOP)\n'
         '→ [B, 56, 4096]',
         BOX_STYLE_HEAD, 1560, 60, BOX_W, 60)

add_edge(r, "e_extract", "llm", "extract",
         "last_hidden_states\n[:, -57:-1, :]")

# ---- L1 Regression Path ----
add_cell(r, "l1_head",
         '<b>L1RegressionActionHead</b>\n'
         '─────────────────────────\n'
         'Reshape: [B, 56, 4096]\n'
         '  → [B, 8, 56×4096]\n'
         '  = [B, 8, 28672]\n'
         '─────────────────────────\n'
         'MLPResNet (2 blocks):\n'
         '  fc1: 28672→4096\n'
         '  ReLU\n'
         '  2× ResNetBlock(4096)\n'
         '  fc2: 4096→7\n'
         '→ [B, 8, 7]\n'
         '─────────────────────────\n'
         'Loss: L1Loss(pred, gt)',
         BOX_STYLE_HEAD, 1560, 160, BOX_W+20, BOX_H+90)

add_edge(r, "e_l1", "extract", "l1_head", "L1 Regression")

# ---- Diffusion Path ----
add_cell(r, "diff_head",
         '<b>DiffusionActionHead</b> (optional)\n'
         '─────────────────────────\n'
         'DDIM Scheduler (50 steps)\n'
         'NoisePredictor:\n'
         '  MLPResNet(2 blocks)\n'
         '  fc1: 28672→4096\n'
         '  fc2: 4096→7\n'
         'SinusoidalTimeEncoding(4096)\n'
         '─────────────────────────\n'
         'Reverse diffusion:\n'
         '  noise → x_T → ... → x_0\n'
         'Loss: MSELoss(noise_pred, noise)',
         BOX_STYLE_HEAD, 1560, 310, BOX_W+20, BOX_H+100)

add_edge(r, "e_diff", "extract", "diff_head", "Diffusion")

# ---- Output ----
add_cell(r, "output",
         '<b>Action Chunk</b>\n'
         '8 steps × 7 DoF\n'
         '─────────────────\n'
         'Δx, Δy, Δz (3)\n'
         'Δroll, Δpitch, Δyaw (3)\n'
         'Gripper open/close (1)\n'
         '─────────────────\n'
         '→ Execute in open-loop\n'
         '   or closed-loop',
         BOX_STYLE, 1560, 460, BOX_W+20, 100)

add_edge(r, "eo1", "l1_head", "output", "unnormalize → execute")
add_edge(r, "eo2", "diff_head", "output", "unnormalize → execute")

# ==============================================================================
# DATA FLOW LEGEND (right side bottom)
# ==============================================================================
add_cell(r, "legend",
         '<b>Legend: Feature Dimensions</b>\n'
         '[B, H, W, C] = [Batch, Height, Width, Channels]\n'
         '[B, S, D] = [Batch, Sequence, Hidden Dim]\n'
         '[B, N, D] = [Batch, Num Patches, Hidden Dim]\n\n'
         '<b>Color Code:</b>\n'
         '🟦 Blue: Preprocessing / Assembly\n'
         '🟩 Green: Vision Backbones\n'
         '🟨 Yellow: Projection Layers\n'
         '🟥 Red: LLM Backbone\n'
         '🟪 Purple: Action Head / Output\n'
         '🟧 Orange: Auxiliary Modules',
         "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#999999;"
         "fontFamily=Consolas;fontSize=10;align=left;verticalAlign=top;",
         80, 650, 310, 160)

# ==============================================================================
# Save
# ==============================================================================
output_path = "openvla_oft_architecture.drawio"
xml_str = minidom.parseString(ET.tostring(root, encoding="unicode")).toprettyxml(indent="  ")

with open(output_path, "w", encoding="utf-8") as f:
    f.write(xml_str)

print(f"Draw.io diagram saved to: {output_path}")
print("Open this file in VS Code (with Draw.io extension) or at https://app.diagrams.net/")
