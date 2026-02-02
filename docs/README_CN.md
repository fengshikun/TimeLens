# TimeLens 文档中心

欢迎来到 TimeLens 中文文档中心！这里提供了关于 TimeLens 项目的详细技术文档。

## 📖 文档索引

### 时间戳编码相关

1. **[Interleaved Textual Timestamp Encoding 实现详解](./interleaved_textual_timestamp_encoding.md)** 🔥
   - 详细解释 TimeLens-7B 中 Interleaved Textual Timestamp Encoding 的实现原理
   - 包含完整的代码示例和数据流程图
   - 说明实现位置和关键代码
   
2. **[时间戳编码示例](./timestamp_encoding_examples.md)**
   - TimeLens-7B（Qwen2.5-VL）的 Interleaved Textual Timestamps 方式
   - TimeLens-8B（Qwen3-VL）的 Video Metadata 方式
   - 不同编码策略的对比和使用场景

## 🔍 快速查找

### 常见问题

**Q: Interleaved Textual Timestamp Encoding 是怎么实现的？**
- A: 查看 [Interleaved Textual Timestamp Encoding 实现详解](./interleaved_textual_timestamp_encoding.md)

**Q: 不同 TimeLens 模型使用什么时间戳编码方式？**
- A: 
  - TimeLens-7B: Interleaved Textual Timestamp Encoding
  - TimeLens-8B: Video Metadata 方式
  - 详见 [时间戳编码示例](./timestamp_encoding_examples.md)

**Q: 如何在代码中使用时间戳编码？**
- A: 参考各文档中的完整代码示例

## 📝 实现位置速查

### Interleaved Textual Timestamp Encoding

| 组件 | 文件路径 | 说明 |
|------|---------|------|
| 核心实现 | `evaluation/utils.py` (第 61-70 行) | 实际调用位置 |
| Prompt 模板 | `evaluation/utils.py` (第 14-17 行) | 特殊的时间戳提示词 |
| 数据集类 | `timelens/dataset/timelens_data.py` | 完整实现 |
| 详细文档 | `docs/timestamp_encoding_examples.md` (第 2 节) | 使用示例 |

## 🚀 快速开始

如果你想快速了解如何使用 TimeLens 模型进行时间戳编码，建议按以下顺序阅读：

1. 先阅读主 [README.md](../README.md) 了解项目概况
2. 查看 [时间戳编码示例](./timestamp_encoding_examples.md) 了解不同模型的编码方式
3. 深入阅读 [Interleaved Textual Timestamp Encoding 实现详解](./interleaved_textual_timestamp_encoding.md) 理解实现细节

## 💡 贡献文档

如果你发现文档有任何问题或需要改进的地方，欢迎：
- 提交 Issue
- 发起 Pull Request
- 在项目中反馈

---

*最后更新: 2026-02-01*
