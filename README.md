# astrbot_plugin_forward_context_layers

在 `on_llm_request` 阶段，递归展开 QQ/OneBot 场景下的多层 `reply + forward`，把每层新增信息注入到 LLM 上下文，减少“只看到外层引用、看不到底层正文”的问题。

## 功能
- 解析多层 `Reply`（可配置最大层数）
- 解析多层 `Forward`（可配置最大层数）
- 将每层内容按 `Layer N` 结构注入 `system_prompt`
- 支持字符截断和总长度上限，避免上下文爆炸

## 指令
- `/fctx` 查看当前运行配置摘要

## 配置项
- `enabled`: 启用插件
- `only_when_replied`: 仅在消息含引用/转发时生效
- `max_reply_layers`: reply 最大解析层数
- `max_forward_layers`: forward 最大解析层数
- `max_chars_per_layer`: 每层最大字符数
- `max_total_chars`: 总注入上限
- `include_sender`: 每层是否附带发送者
- `include_non_text_tokens`: 是否注入图片/表情占位
- `include_current_outline`: 是否注入当前消息概要层

## 安装
将目录放到 AstrBot 插件目录并重启。

GitHub: https://github.com/constansino/astrbot_plugin_forward_context_layers
