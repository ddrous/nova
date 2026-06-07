#%%
"""
Export JAX/Equinox NOVA model blocks to ONNX, then statically quantize.

KEY FIX — Gemm rank-1 B input
──────────────────────────────
jax2onnx maps JAX's lax.dot_general to ONNX Gemm.  Gemm requires both A and
B to be rank-2 matrices.  When eqx.nn.GRUCell (or any Linear applied to a
1-D vector) is traced, the *data* vector (hidden state h, input x …) becomes
the second Gemm operand at rank-1, causing:

  [ShapeInferenceError] Second input does not have rank 2

ONNX MatMul handles a 1-D second argument correctly per spec:
  "If the second argument is 1-D, it is promoted to a matrix by appending a 1
   to its dimensions. After matrix multiplication the appended 1 is removed."
i.e.  MatMul([M, K], [K]) → [M]  ✓

We therefore replace every Gemm(A, B_rank1, C) with  MatMul(A, B) + Add(C).
This fix is applied to *all* exported models for safety; it is a no-op for
models that already use MatMul or have rank-2 Gemm operands.
"""

import os
import glob
import shutil

import jax
import jax.numpy as jnp
import numpy as np

import onnx
import onnx.numpy_helper as numpy_helper
import onnx.helper as onnx_helper
from onnx import shape_inference as onnx_shape_inference
from jax2onnx import to_onnx

import onnxruntime as ort
from onnxruntime.quantization import quantize_static, QuantType, CalibrationDataReader

# ── quant_pre_process import (API differs across ORT versions) ────────────────
try:
    from onnxruntime.quantization import shape_inference as ort_si
    _quant_pre_process = ort_si.quant_pre_process
except AttributeError:
    try:
        from onnxruntime.quantization.quant_pre_process import quant_pre_process as _quant_pre_process
    except ImportError:
        _quant_pre_process = None

from src.config import CONFIG
from src.utils import get_model, load_weights


# ══════════════════════════════════════════════════════════════════════════════
# ONNX GRAPH SURGERY — fix rank-1 Gemm B inputs
# ══════════════════════════════════════════════════════════════════════════════

def fix_gemm_rank1_inputs(model: onnx.ModelProto) -> tuple:
    """
    Walk every Gemm node in the graph.  If the second input (B) is rank-1,
    replace the node with  MatMul(A, B) + optional Add(bias).

    Returns (fixed_model, n_nodes_replaced).
    """
    # ── Populate shape tables ────────────────────────────────────────────────
    try:
        model = onnx_shape_inference.infer_shapes(model)
    except Exception:
        pass  # best-effort — we still try with whatever info we have

    init_shapes: dict[str, list] = {
        init.name: list(init.dims) for init in model.graph.initializer
    }
    vi_shapes: dict[str, list] = {}
    for vi in (list(model.graph.input)
               + list(model.graph.value_info)
               + list(model.graph.output)):
        try:
            if vi.type.HasField("tensor_type") and vi.type.tensor_type.HasField("shape"):
                vi_shapes[vi.name] = [d.dim_value
                                      for d in vi.type.tensor_type.shape.dim]
        except Exception:
            pass

    new_nodes:   list = []
    extra_inits: list = []
    n_fixed = 0

    for node in model.graph.node:
        # Only touch Gemm nodes with at least 2 inputs
        if node.op_type != "Gemm" or len(node.input) < 2:
            new_nodes.append(node)
            continue

        b_name = node.input[1]

        # Determine B's rank
        if b_name in init_shapes:
            b_rank = len(init_shapes[b_name])
        elif b_name in vi_shapes:
            b_rank = len(vi_shapes[b_name])
        else:
            # Cannot determine rank — leave the node as-is
            new_nodes.append(node)
            continue

        if b_rank != 1:
            new_nodes.append(node)   # rank-2 B is fine for Gemm
            continue

        # ── Parse Gemm attributes ────────────────────────────────────────────
        alpha, beta, transA = 1.0, 1.0, 0
        for attr in node.attribute:
            if   attr.name == "alpha":  alpha  = float(attr.f)
            elif attr.name == "beta":   beta   = float(attr.f)
            elif attr.name == "transA": transA = int(attr.i)
            # transB is a no-op for rank-1 B — ignored

        uid     = f"gemfix_{n_fixed}"
        a_name  = node.input[0]
        c_name  = (node.input[2]
                   if len(node.input) > 2 and node.input[2]
                   else None)               # empty string ⇒ no bias
        out_name = node.output[0]

        curr_a = a_name

        # ── Optional A transpose ─────────────────────────────────────────────
        if transA:
            a_t = f"{uid}_A_T"
            new_nodes.append(onnx_helper.make_node(
                "Transpose", [curr_a], [a_t], name=f"{uid}_transA"))
            curr_a = a_t

        # ── MatMul: replaces the matrix multiply part of Gemm ────────────────
        # MatMul([M, K], [K])  →  [M]   (ONNX spec handles rank-1 right arg)
        mm_out = f"{uid}_mm"
        new_nodes.append(onnx_helper.make_node(
            "MatMul", [curr_a, b_name], [mm_out], name=f"{uid}_matmul"))
        curr_out = mm_out

        # ── Optional alpha scaling ───────────────────────────────────────────
        if abs(alpha - 1.0) > 1e-7:
            alpha_init_name = f"{uid}_alpha"
            extra_inits.append(numpy_helper.from_array(
                np.array(alpha, dtype=np.float32), name=alpha_init_name))
            scaled = f"{uid}_alpha_out"
            new_nodes.append(onnx_helper.make_node(
                "Mul", [curr_out, alpha_init_name], [scaled],
                name=f"{uid}_alpha_mul"))
            curr_out = scaled

        # ── Optional bias addition (C) ───────────────────────────────────────
        if c_name:
            bias = c_name
            if abs(beta - 1.0) > 1e-7:
                beta_init_name = f"{uid}_beta"
                extra_inits.append(numpy_helper.from_array(
                    np.array(beta, dtype=np.float32), name=beta_init_name))
                scaled_bias = f"{uid}_scaled_bias"
                new_nodes.append(onnx_helper.make_node(
                    "Mul", [c_name, beta_init_name], [scaled_bias],
                    name=f"{uid}_beta_mul"))
                bias = scaled_bias
            new_nodes.append(onnx_helper.make_node(
                "Add", [curr_out, bias], [out_name], name=f"{uid}_add_bias"))
        else:
            new_nodes.append(onnx_helper.make_node(
                "Identity", [curr_out], [out_name], name=f"{uid}_identity"))

        n_fixed += 1

    if n_fixed:
        del model.graph.node[:]
        model.graph.node.extend(new_nodes)
        model.graph.initializer.extend(extra_inits)

    return model, n_fixed


# ══════════════════════════════════════════════════════════════════════════════
# EXPORT + FIX HELPER
# ══════════════════════════════════════════════════════════════════════════════

def export_and_fix(fn, dummy_inputs, opset: int, name: str, out_dir: str) -> str:
    """
    1. Trace fn with jax2onnx.
    2. Apply Gemm rank-1-B fix.
    3. Run onnx.checker + quick ORT load.
    4. Save to {out_dir}/{name}.onnx and return the path.
    """
    fp32_path = os.path.join(out_dir, f"{name}.onnx")

    model_proto = to_onnx(fn, dummy_inputs, opset=opset)
    model_proto, n_fixed = fix_gemm_rank1_inputs(model_proto)

    if n_fixed:
        print(f"    ✔  {n_fixed} Gemm(rank-1 B) → MatMul+Add replacement(s) in '{name}'")
    else:
        print(f"    –  no rank-1 Gemm found in '{name}'")

    try:
        onnx.checker.check_model(model_proto)
    except onnx.checker.ValidationError as e:
        print(f"    ⚠️  onnx.checker warning for '{name}': {e}")

    onnx.save(model_proto, fp32_path)

    # Quick native-ORT load to catch any remaining issues before quantizing
    try:
        ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
        print(f"    ✓  ORT load OK: {name}.onnx")
    except Exception as e:
        print(f"    ✗  ORT load FAILED for {name}.onnx: {e}")
        print(f"       ↳ The quantized version will also fail. Check the model.")

    return fp32_path


# ══════════════════════════════════════════════════════════════════════════════
# CALIBRATION DATA READER
# ══════════════════════════════════════════════════════════════════════════════

class ModelCalibrationDataReader(CalibrationDataReader):
    """Provides random float32 calibration data for static quantization."""

    def __init__(self, model_path: str, num_samples: int = 10):
        model = onnx.load(model_path)
        self.inputs: list[tuple[str, tuple]] = []
        for inp in model.graph.input:
            name = inp.name
            # Skip quantization-internal tensors from a prior run
            if any(s in name for s in ["_reduce_axis", "_zero_point", "_scale"]):
                continue
            shape = tuple(
                dim.dim_value if dim.dim_value > 0 else 1
                for dim in inp.type.tensor_type.shape.dim
            )
            self.inputs.append((name, shape))
        self.num_samples = num_samples
        self.iter = 0

    def get_next(self):
        if self.iter >= self.num_samples:
            return None
        feed = {}
        for name, shape in self.inputs:
            lo, hi = (-1.0, 1.0) if "coord" in name else (0.0, 1.0)
            feed[name] = np.random.uniform(lo, hi, shape).astype(np.float32)
        self.iter += 1
        return feed


# ══════════════════════════════════════════════════════════════════════════════
# STATIC QUANTIZATION
# ══════════════════════════════════════════════════════════════════════════════

def quantize_model_static(fp32_path: str, quant_path: str,
                          calib_samples: int = 10) -> bool:
    """
    Attempt static INT8 quantization.  Falls back to copying the FP32 model
    if anything goes wrong, so the HTML can always load *_quant.onnx.
    """
    model_input = fp32_path

    # Pre-process: add shape info expected by the quantizer
    if _quant_pre_process is not None:
        pre_path = fp32_path.replace(".onnx", "_pre.onnx")
        try:
            _quant_pre_process(fp32_path, pre_path, skip_symbolic_shape=True)
            model_input = pre_path
        except Exception as e:
            print(f"    ⚠️  quant_pre_process skipped ({e})")
    else:
        print("    ⚠️  quant_pre_process not found in this ORT version — skipping")

    calib_reader = ModelCalibrationDataReader(fp32_path, num_samples=calib_samples)

    try:
        quantize_static(
            model_input=model_input,
            model_output=quant_path,
            calibration_data_reader=calib_reader,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QInt8,
            extra_options={"ActivationSymmetric": True, "WeightSymmetric": True},
        )
        # Verify the quantized model loads natively before declaring success
        ort.InferenceSession(quant_path, providers=["CPUExecutionProvider"])

        sz0 = os.path.getsize(fp32_path)  / (1024 * 1024)
        sz1 = os.path.getsize(quant_path) / (1024 * 1024)
        print(f"    ✓  {os.path.basename(fp32_path)}: {sz0:.1f} MB → {sz1:.1f} MB")
        return True

    except Exception as e:
        print(f"    ❌ Quantization failed: {e}")
        print(f"       Copying FP32 model as fallback (still loadable in browser).")
        shutil.copy2(fp32_path, quant_path)
        return False

    finally:
        if model_input != fp32_path and os.path.exists(model_input):
            os.remove(model_input)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def export_all():
    print("Initializing model …")
    key   = jax.random.PRNGKey(0)
    model = get_model(key)

    # Opset 17: broad ORT-Web coverage without requiring opset-18+ ops
    OPSET = 17

    # ── Load weights ─────────────────────────────────────────────────────────
    PATTERN = ("../../HiddenRep/latent_action_models/experiments/"
               "260312-113749-PerfectMNIST*/artefacts/model_phase2_final.eqx")
    matches = glob.glob(PATTERN)
    if not matches:
        print("⚠️  No checkpoint found — exporting with random weights (debug only).")
    else:
        MODEL_PATH = sorted(matches)[0]
        print(f"Loading weights from: {MODEL_PATH}")
        model = load_weights(model, MODEL_PATH)

    out_dir = "models"
    os.makedirs(out_dir, exist_ok=True)

    H, W, C = CONFIG["frame_shape"]          # (64, 64, 1) for Moving MNIST
    dummy_img    = jnp.zeros((C, H, W),             dtype=jnp.float32)
    dummy_z      = jnp.zeros((model.d_theta,),       dtype=jnp.float32)
    dummy_u      = jnp.zeros((CONFIG["lam_space"],), dtype=jnp.float32)
    dummy_m      = jnp.zeros((CONFIG["mem_space"],), dtype=jnp.float32)
    dummy_coords = jnp.zeros((H, W, 3),              dtype=jnp.float32)

    # ── Export FP32 + apply Gemm fix ─────────────────────────────────────────
    print("\n📦 Exporting FP32 models (Gemm rank-1 fix applied to each)…")

    print("  1/5  Encoder          img[C,H,W] → z[d_theta]")
    export_and_fix(
        lambda img: model.encoder(img),
        [dummy_img], OPSET, "encoder", out_dir)

    print("  2/5  FDM              z,u → z_next")
    export_and_fix(
        lambda z, u: model.forward_dyn(z, u),
        [dummy_z, dummy_u], OPSET, "fdm", out_dir)

    print("  3/5  IDM              z,z_next → u")
    export_and_fix(
        lambda z, zn: model.action_model.inverse_dynamics(z, zn),
        [dummy_z, dummy_z], OPSET, "idm", out_dir)

    print("  4/5  GCM step         m,z,u → (u_next, m_next)")
    def gcm_step(m, z, u):
        # step_idx=0 is a static constant (unused in GRU path)
        m_next = model.action_model.encode_memory(m, 0, z, u)
        u_next = model.action_model.decode_memory(m_next, 0, z)
        return u_next, m_next
    export_and_fix(gcm_step,
        [dummy_m, dummy_z, dummy_u], OPSET, "gcm_step", out_dir)

    print("  5/5  INR Renderer     z,coords[H,W,3] → pixels[H,W,C]")
    export_and_fix(
        lambda z, coords: model.render_frame(z, coords),
        [dummy_z, dummy_coords], OPSET, "renderer", out_dir)

    print(f"\n✅  FP32 models saved to '{out_dir}/'")

    # ── Static quantization ───────────────────────────────────────────────────
    print("\n⚡ Quantizing (static INT8 Q/DQ) …")
    for name in ["encoder", "fdm", "idm", "gcm_step", "renderer"]:
        print(f"  {name} …")
        quantize_model_static(
            fp32_path  = os.path.join(out_dir, f"{name}.onnx"),
            quant_path = os.path.join(out_dir, f"{name}_quant.onnx"),
        )

    print("\n🚀 Done!  *_quant.onnx files are ready for the browser.")
    print("   Place index.html, models/, and assets/ in the same folder.")
    print("   Serve with:  python -m http.server 8000")


if __name__ == "__main__":
    export_all()
