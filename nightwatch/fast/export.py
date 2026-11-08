"""Export the fast head (decode included) to ONNX, build a TensorRT engine, measure latency.

Falls back to ONNX Runtime CUDA if TensorRT is unavailable. Results -> runs/<name>/deploy.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from nightwatch.fast.data import FH, FW
from nightwatch.fast.model import CenterNetRGBT, Exportable
from nightwatch.fast.train import IN_CH


def bench(fn, n=300, warm=30):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--backbone", default="resnet18.a1_in1k")
    a = ap.parse_args()
    d = f"runs/{a.name}"
    summ = json.load(open(f"{d}/train_summary.json"))
    cin = IN_CH[summ["modal"]]
    net = CenterNetRGBT(a.backbone, cin, pretrained=False)
    net.load_state_dict(torch.load(f"{d}/best.pt", map_location="cpu"))
    m = Exportable(net.eval()).cuda().eval()
    x = torch.randn(1, cin, FH, FW, device="cuda")
    res = {"input": [1, cin, FH, FW], "gpu": torch.cuda.get_device_name()}

    with torch.no_grad():
        res["torch_fp32_ms"] = bench(lambda: m(x))
        with torch.autocast("cuda", dtype=torch.float16):
            res["torch_fp16_ms"] = bench(lambda: m(x))

    onnx_path = f"{d}/fast_head.onnx"
    torch.onnx.export(m, (x,), onnx_path, input_names=["sensors"], output_names=["boxes", "scores"], opset_version=17,
                      dynamo=False)
    res["onnx_mb"] = os.path.getsize(onnx_path) / 1e6

    try:
        import tensorrt as trt
        res["tensorrt_version"] = trt.__version__
        # TensorRT >= 10.12/11 is strongly typed: precision comes from the ONNX graph, so export an FP16 graph.
        strongly_typed = not hasattr(trt.BuilderFlag, "FP16")
        trt_onnx = onnx_path
        x_in = x
        if strongly_typed:
            m16 = Exportable(CenterNetRGBT(a.backbone, cin, pretrained=False)).cuda().eval()
            m16.net.load_state_dict(net.state_dict())
            m16 = m16.half()
            x_in = x.half()
            trt_onnx = f"{d}/fast_head_fp16.onnx"
            torch.onnx.export(m16, (x_in,), trt_onnx, input_names=["sensors"], output_names=["boxes", "scores"],
                              opset_version=17, dynamo=False)
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        flags = (1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)) if strongly_typed else 0
        network = builder.create_network(flags)
        parser = trt.OnnxParser(network, logger)
        assert parser.parse(open(trt_onnx, "rb").read()), [parser.get_error(i) for i in range(parser.num_errors)]
        cfg = builder.create_builder_config()
        cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
        if not strongly_typed:
            cfg.set_flag(trt.BuilderFlag.FP16)
        eng_bytes = builder.build_serialized_network(network, cfg)
        assert eng_bytes is not None, "engine build failed"
        open(f"{d}/fast_head_fp16.engine", "wb").write(eng_bytes)
        rt = trt.Runtime(logger)
        eng = rt.deserialize_cuda_engine(eng_bytes)
        ctx = eng.create_execution_context()
        tdt = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16, trt.DataType.INT32: torch.int32,
               trt.DataType.INT64: torch.int64}
        outs = {}
        for i in range(eng.num_io_tensors):          # allocate exactly what the engine says (dtype + shape)
            name = eng.get_tensor_name(i)
            if eng.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                outs[name] = torch.empty(tuple(eng.get_tensor_shape(name)), device="cuda",
                                         dtype=tdt[eng.get_tensor_dtype(name)])
        in_dt = tdt[eng.get_tensor_dtype("sensors")]
        try:   # a real test frame, not noise, for the equivalence check
            from nightwatch.fast.data import prep_eval
            xr = torch.from_numpy(np.load("data/cache_fast/test.npy", mmap_mode="r")[:1].copy()).cuda()
            x = prep_eval(xr, summ["modal"]).contiguous()
        except Exception:
            pass
        x_in = x.to(in_dt).contiguous()
        ctx.set_tensor_address("sensors", x_in.data_ptr())
        for k, v in outs.items():
            ctx.set_tensor_address(k, v.data_ptr())
        stream = torch.cuda.Stream()
        res["tensorrt_fp16_ms"] = bench(lambda: ctx.execute_async_v3(stream.cuda_stream))
        with torch.no_grad():
            ref_b, ref_s = m(x)
        ctx.execute_async_v3(stream.cuda_stream); stream.synchronize()
        res["tensorrt_max_score_diff"] = float((outs["scores"].float() - ref_s).abs().max())
        res["engine_mb"] = eng_bytes.nbytes / 1e6
    except Exception as e:
        res["tensorrt_error"] = repr(e)[:300]
        try:
            import onnxruntime as ort
            s = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider"])
            xn = x.cpu().numpy()
            res["onnxruntime_cuda_ms"] = bench(lambda: s.run(None, {"sensors": xn}), n=100) if s else None
        except Exception as e2:
            res["onnxruntime_error"] = repr(e2)[:300]
    for k in [k for k in res if k.endswith("_ms")]:
        res[k.replace("_ms", "_fps")] = 1000 / res[k]
    json.dump(res, open(f"{d}/deploy.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
