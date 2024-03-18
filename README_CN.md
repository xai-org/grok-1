# grok-1

[English Document](./README.md) **Chinese Document**

该存储库包含用于加载和运行 `Grok-1` 开放权重模型的 `JAX` 示例代码。

确保下载检查点并将 `ckpt-0` 目录放置在 `checkpoints` 中。
然后，运行

```shell
pip install -r requirements.txt
python run.py
```

测试代码。

该脚本在测试输入上加载模型中的检查点和样本。

由于模型规模较大（314B参数），需要有足够GPU内存的机器才能使用示例代码测试模型。
该存储库中 `MoE 层` 的实现效率不高。选择该实现是为了避免需要自定义内核来验证模型的正确性。

## 下载权重

您可以使用 `torrent` 客户端和此磁力链接下载权重：

```shell
magnet:?xt=urn:btih:5f96d43576e3d386c9ba65b883210a393b68210e&tr=https%3A%2F%2Facademictorrents.com%2Fannounce.php&tr=udp%3A%2F%2Ftracker.coppersurfer.tk%3A6969&tr=udp%3A%2F%2Ftracker.opentrackr.org%3A1337%2Fannounce
```

## 执照

此版本中的代码和相关 `Grok-1` 权重已获得许可 `Apache 2.0` 许可证。
该许可证仅适用于本文件中的源文件 `Grok-1` 的存储库和模型权重。
