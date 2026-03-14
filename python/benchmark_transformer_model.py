#!/usr/bin/env python3
"""Benchmark a randomly-initialized transformer model via KataGo C++ MCTS.

Generates a random transformer model in .kgtr.gz format, then invokes the
compiled katago benchmark command for MCTS+CUDA inference benchmarking.

Usage:
    python benchmark_transformer_model.py \
        --num-layers 24 --hidden-size 1024 --num-heads 16 \
        --katago /path/to/katago --config gtp.cfg

    # With extra args passed through to katago benchmark:
    python benchmark_transformer_model.py \
        --num-layers 12 --hidden-size 384 --num-heads 6 \
        --katago ./katago --config gtp.cfg -t 4,8,16

    # Keep the generated model file:
    python benchmark_transformer_model.py \
        --num-layers 12 --hidden-size 192 --num-heads 3 \
        --katago ./katago --config gtp.cfg \
        --output my_model.kgtr.gz --keep-model
"""

import argparse
import os
import subprocess
import sys
import tempfile

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_NANO_PATH = os.path.normpath(
    os.path.join(_SCRIPT_DIR, "..", "..", "KataGo_Transformer", "nano")
)


def _setup_nano_imports(nano_path):
    if not os.path.isdir(nano_path):
        print(f"错误: nano 目录不存在: {nano_path}", file=sys.stderr)
        print("请用 --nano-path 指定 KataGo_Transformer/nano 目录路径", file=sys.stderr)
        sys.exit(1)
    sys.path.insert(0, nano_path)


def _create_random_model(num_layers, hidden_size, num_heads, ffn_dim, pos_len, score_mode):
    from configs import make_config
    from model import Model

    config = make_config(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
    )
    model = Model(config, pos_len=pos_len, score_mode=score_mode)
    model.initialize()
    model.eval()
    return model, config


def _export_model(model, config, pos_len, output_path):
    from configs import get_num_bin_input_features, get_num_global_input_features
    from export_cuda import (
        MAGIC,
        FORMAT_VERSION,
        _collect_tensors,
        _open_output,
        _score_mode_id,
        _write_f32,
        _write_i32,
        _write_string,
        _write_tensor,
        _write_u32,
    )

    tensors = _collect_tensors(model)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with _open_output(output_path) as out:
        out.write(MAGIC)
        _write_u32(out, FORMAT_VERSION)
        _write_string(out, os.path.splitext(os.path.basename(output_path))[0])
        _write_i32(out, config["version"])
        _write_i32(out, pos_len)
        _write_i32(out, config["hidden_size"])
        _write_i32(out, config["num_layers"])
        _write_i32(out, config["num_heads"])
        _write_i32(out, config["ffn_dim"])
        _write_i32(out, get_num_bin_input_features(config))
        _write_i32(out, get_num_global_input_features(config))
        _write_i32(out, {"cnn1": 1, "cnn3": 3, "cnn5": 5}[model.stem])
        _write_i32(out, 0)  # APE flag (always disabled)
        _write_i32(out, _score_mode_id(model.value_head.score_mode))
        _write_i32(out, model.value_head.num_scorebeliefs)
        _write_i32(out, model.value_head.scorebelief_len)

        # Post-process params (fixed values)
        _write_f32(out, 20.0)   # tdScoreMultiplier
        _write_f32(out, 20.0)   # scoreMeanMultiplier
        _write_f32(out, 20.0)   # scoreStdevMultiplier
        _write_f32(out, 20.0)   # leadMultiplier
        _write_f32(out, 40.0)   # varianceTimeMultiplier
        _write_f32(out, 0.25)   # shorttermValueErrorMultiplier
        _write_f32(out, 150.0)  # shorttermScoreErrorMultiplier
        _write_f32(out, 1.0)    # outputScaleMultiplier

        _write_u32(out, len(tensors))
        for name, tensor in tensors:
            _write_tensor(out, name, tensor)


def main():
    parser = argparse.ArgumentParser(
        description="生成随机 Transformer 模型并运行 KataGo C++ MCTS benchmark",
        epilog="所有未识别的参数将透传给 katago benchmark（如 -v, -t, -s 等）",
    )
    parser.add_argument("--num-layers", type=int, required=True, help="Transformer 层数")
    parser.add_argument("--hidden-size", type=int, required=True, help="隐藏维度")
    parser.add_argument("--num-heads", type=int, required=True, help="注意力头数")
    parser.add_argument("--katago", type=str, required=True, help="katago 可执行文件路径")
    parser.add_argument("--config", type=str, required=True, help="GTP 配置文件路径")
    parser.add_argument("--ffn-dim", type=int, default=None,
                        help="SwiGLU FFN 中间维度（默认 hidden_size * 8 // 3）")
    parser.add_argument("--pos-len", type=int, default=19, help="棋盘边长（默认 19）")
    parser.add_argument("--score-mode", type=str, default="mixop",
                        choices=["mixop", "mix", "simple"],
                        help="Value head 模式（默认 mixop）")
    parser.add_argument("--nano-path", type=str, default=None,
                        help="KataGo_Transformer/nano 目录路径（默认自动推断）")
    parser.add_argument("--output", type=str, default=None,
                        help="导出的模型文件路径（默认使用临时文件）")
    parser.add_argument("--keep-model", action="store_true",
                        help="不删除生成的模型文件")

    args, extra_args = parser.parse_known_args()

    nano_path = args.nano_path or _DEFAULT_NANO_PATH
    _setup_nano_imports(nano_path)

    # Validate
    if args.hidden_size % args.num_heads != 0:
        print(f"错误: hidden_size ({args.hidden_size}) 必须能被 num_heads ({args.num_heads}) 整除",
              file=sys.stderr)
        sys.exit(1)

    ffn_dim = args.ffn_dim if args.ffn_dim is not None else args.hidden_size * 8 // 3
    model_name = f"b{args.num_layers}c{args.hidden_size}h{args.num_heads}"

    print("=" * 60)
    print("Transformer Random Model MCTS Benchmark")
    print("=" * 60)
    print(f"架构: layers={args.num_layers}, hidden={args.hidden_size}, "
          f"heads={args.num_heads}, ffn={ffn_dim}")
    print(f"Score mode: {args.score_mode}")
    print(f"Board: {args.pos_len}x{args.pos_len}")
    print()

    # Step 1: Create random model
    print("步骤 1/3: 创建随机初始化模型...")
    model, config = _create_random_model(
        args.num_layers, args.hidden_size, args.num_heads,
        args.ffn_dim, args.pos_len, args.score_mode,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {total_params:,}")

    # Step 2: Export to .kgtr.gz
    use_temp = args.output is None
    if use_temp:
        tmp = tempfile.NamedTemporaryFile(
            prefix=f"benchmark_{model_name}_", suffix=".kgtr.gz", delete=False
        )
        output_path = tmp.name
        tmp.close()
    else:
        output_path = args.output

    print(f"步骤 2/3: 导出模型到 {output_path} ...")
    _export_model(model, config, args.pos_len, output_path)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  文件大小: {file_size_mb:.1f} MB")

    # Free model memory before launching C++ benchmark
    del model

    # Step 3: Run katago benchmark
    print(f"步骤 3/3: 运行 katago benchmark ...")
    print("=" * 60)
    print()

    cmd = [args.katago, "benchmark", "-model", output_path, "-config", args.config]
    cmd.extend(extra_args)

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print(f"\n错误: 找不到 katago 可执行文件: {args.katago}", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"\nkatago benchmark 返回错误码: {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)
    finally:
        if use_temp and not args.keep_model:
            try:
                os.unlink(output_path)
            except OSError:
                pass
        elif args.keep_model or not use_temp:
            print(f"\n模型文件已保留: {output_path}")


if __name__ == "__main__":
    main()
