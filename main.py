import json
from dataclasses import dataclass
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import At, Face, Forward, Image, Node, Nodes, Plain, Reply
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, register


@dataclass
class LayerView:
    level: int
    text: str
    sender: str = ""


@register(
    "astrbot_plugin_forward_context_layers",
    "constansino",
    "Reply/Forward 多层上下文展开注入",
    "v1.0.0",
)
class ForwardContextLayersPlugin(Star):
    """将多层 reply + forward 展开为结构化上下文，注入到 LLM 请求。"""

    MARKER = "<!-- FORWARD_CONTEXT_LAYERS_V1 -->"

    def __init__(self, context: Context, config: dict | None = None):
        super().__init__(context)
        self.config = config or {}

    @property
    def enabled(self) -> bool:
        return bool(self.config.get("enabled", True))

    @property
    def max_reply_layers(self) -> int:
        return max(0, int(self.config.get("max_reply_layers", 5)))

    @property
    def max_forward_layers(self) -> int:
        return max(0, int(self.config.get("max_forward_layers", 5)))

    @property
    def max_chars_per_layer(self) -> int:
        return max(100, int(self.config.get("max_chars_per_layer", 900)))

    @property
    def max_total_chars(self) -> int:
        return max(500, int(self.config.get("max_total_chars", 5000)))

    @property
    def include_sender(self) -> bool:
        return bool(self.config.get("include_sender", True))

    @property
    def include_non_text_tokens(self) -> bool:
        return bool(self.config.get("include_non_text_tokens", True))

    @property
    def include_current_outline(self) -> bool:
        return bool(self.config.get("include_current_outline", True))

    @property
    def only_when_replied(self) -> bool:
        return bool(self.config.get("only_when_replied", True))

    @filter.command("fctx")
    async def fctx(self, event: AstrMessageEvent):
        msg = (
            "Forward Context Layers 插件\n"
            f"- enabled: {self.enabled}\n"
            f"- max_reply_layers: {self.max_reply_layers}\n"
            f"- max_forward_layers: {self.max_forward_layers}\n"
            f"- max_chars_per_layer: {self.max_chars_per_layer}\n"
            f"- max_total_chars: {self.max_total_chars}\n"
            f"- only_when_replied: {self.only_when_replied}\n"
            "\n"
            "说明: 本插件在 on_llm_request 阶段注入多层引用/转发上下文。"
        )
        yield event.plain_result(msg)

    @filter.on_llm_request()
    async def inject_layered_context(self, event: AstrMessageEvent, req: ProviderRequest):
        if not self.enabled:
            return

        chain = self._safe_get_messages(event)
        if self.only_when_replied and not self._has_reply_or_forward(chain):
            return

        if hasattr(req, "system_prompt") and isinstance(req.system_prompt, str):
            if self.MARKER in req.system_prompt:
                return

        views = await self._build_layer_views(event, chain)
        if not views:
            return

        injected = self._format_for_prompt(event, views)
        if not injected.strip():
            return

        try:
            req.system_prompt = f"{(req.system_prompt or '').rstrip()}\n\n{injected}"
            logger.info(
                "[fctx] injected layered context: views=%s, chars=%s",
                len(views),
                len(injected),
            )
        except Exception as e:
            logger.warning(f"[fctx] inject failed: {e}")

    async def _build_layer_views(
        self,
        event: AstrMessageEvent,
        current_chain: list[Any],
    ) -> list[LayerView]:
        views: list[LayerView] = []

        if self.include_current_outline:
            current_text = self._chain_to_text(current_chain)
            if current_text.strip():
                views.append(LayerView(level=0, text=current_text, sender="当前消息"))

        reply_segments = [seg for seg in current_chain if isinstance(seg, Reply)]
        for reply_seg in reply_segments:
            nested = await self._expand_reply(event, reply_seg, 1)
            views.extend(nested)

        return self._truncate_views(views)

    async def _expand_reply(
        self,
        event: AstrMessageEvent,
        reply_seg: Reply,
        level: int,
    ) -> list[LayerView]:
        if level > self.max_reply_layers:
            return []

        result: list[LayerView] = []

        chain = getattr(reply_seg, "chain", None) or []
        sender = str(getattr(reply_seg, "sender_nickname", "") or "")
        embedded_text = self._chain_to_text(chain)
        if not embedded_text.strip():
            embedded_text = str(getattr(reply_seg, "message_str", "") or "").strip()

        if embedded_text:
            result.append(LayerView(level=level, text=embedded_text, sender=sender))

        for forward_seg in [seg for seg in chain if isinstance(seg, (Forward, Nodes, Node))]:
            result.extend(await self._expand_forward_component(event, forward_seg, level))

        reply_id = self._get_reply_id(reply_seg)
        if not reply_id:
            return result

        payload = await self._call_get_msg(event, reply_id)
        if not isinstance(payload, dict):
            return result

        sender_name = self._extract_sender_name(payload)
        body_text, forward_ids = self._extract_text_and_forward_ids(payload)

        if body_text:
            result.append(LayerView(level=level, text=body_text, sender=sender_name))

        for fid in forward_ids:
            result.extend(await self._expand_forward_payload(event, fid, level, 1))

        if level < self.max_reply_layers:
            nested_reply_id = self._extract_reply_id_from_payload(payload)
            if nested_reply_id and nested_reply_id != reply_id:
                nested_payload = await self._call_get_msg(event, nested_reply_id)
                nested_seg = self._payload_to_reply_segment(nested_payload)
                if nested_seg:
                    result.extend(await self._expand_reply(event, nested_seg, level + 1))

        return result

    async def _expand_forward_component(
        self,
        event: AstrMessageEvent,
        comp: Any,
        level: int,
    ) -> list[LayerView]:
        if isinstance(comp, Forward):
            fid = str(getattr(comp, "id", "") or "")
            if fid:
                return await self._expand_forward_payload(event, fid, level, 1)
            return []

        if isinstance(comp, Node):
            text = self._chain_to_text(getattr(comp, "content", []) or [])
            sender = str(getattr(comp, "name", "") or getattr(comp, "uin", "") or "")
            return [LayerView(level=level, text=text, sender=sender)] if text else []

        if isinstance(comp, Nodes):
            ret: list[LayerView] = []
            for node in getattr(comp, "nodes", []) or []:
                ret.extend(await self._expand_forward_component(event, node, level))
            return ret

        return []

    async def _expand_forward_payload(
        self,
        event: AstrMessageEvent,
        forward_id: str,
        level: int,
        fdepth: int,
    ) -> list[LayerView]:
        if fdepth > self.max_forward_layers:
            return []

        payload = await self._call_get_forward_msg(event, forward_id)
        if not isinstance(payload, dict):
            return []

        data = self._ob_data(payload)
        nodes = (
            data.get("messages")
            or data.get("message")
            or data.get("nodes")
            or data.get("nodeList")
        )
        if not isinstance(nodes, list):
            return []

        views: list[LayerView] = []
        for node in nodes:
            if not isinstance(node, dict):
                continue

            sender = self._extract_sender_from_node(node)
            content = node.get("message")
            if content is None:
                content = node.get("content", [])

            chain = self._normalize_onebot_chain(content)
            text = self._onebot_chain_to_text(chain)
            if text:
                views.append(LayerView(level=level, text=text, sender=sender))

            nested_fids = self._extract_forward_ids_from_chain(chain)
            for nested_fid in nested_fids:
                views.extend(
                    await self._expand_forward_payload(
                        event,
                        nested_fid,
                        level,
                        fdepth + 1,
                    )
                )

        return views

    def _safe_get_messages(self, event: AstrMessageEvent) -> list[Any]:
        try:
            msgs = event.get_messages()
            if isinstance(msgs, list):
                return msgs
        except Exception:
            pass

        try:
            raw = getattr(event, "message_obj", None)
            chain = getattr(raw, "message", None)
            if isinstance(chain, list):
                return chain
        except Exception:
            pass

        return []

    def _has_reply_or_forward(self, chain: list[Any]) -> bool:
        for seg in chain:
            if isinstance(seg, (Reply, Forward, Node, Nodes)):
                return True
        return False

    def _chain_to_text(self, chain: list[Any]) -> str:
        parts: list[str] = []
        for seg in chain:
            if isinstance(seg, Plain):
                t = str(getattr(seg, "text", "") or "")
                if t:
                    parts.append(t)
            elif isinstance(seg, At):
                qq = getattr(seg, "qq", "")
                name = getattr(seg, "name", "")
                mention = str(name or qq or "")
                if mention:
                    parts.append(f"@{mention}")
            elif isinstance(seg, Reply):
                ms = str(getattr(seg, "message_str", "") or "")
                if ms:
                    parts.append(f"[引用:{ms}]")
            elif isinstance(seg, (Forward, Node, Nodes)):
                parts.append("[转发消息]")
            elif self.include_non_text_tokens and isinstance(seg, Image):
                parts.append("[图片]")
            elif self.include_non_text_tokens and isinstance(seg, Face):
                face_id = getattr(seg, "id", "")
                parts.append(f"[表情:{face_id}]")
        return " ".join([p.strip() for p in parts if p and p.strip()]).strip()

    def _normalize_onebot_chain(self, raw: Any) -> list[dict[str, Any]]:
        if isinstance(raw, list):
            return [x for x in raw if isinstance(x, dict)]

        if isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return [x for x in parsed if isinstance(x, dict)]
            except Exception:
                return [{"type": "text", "data": {"text": raw}}]

        return []

    def _onebot_chain_to_text(self, chain: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for seg in chain:
            seg_type = seg.get("type")
            data = seg.get("data") if isinstance(seg.get("data"), dict) else {}

            if seg_type in ("text", "plain"):
                text = data.get("text", "")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
            elif seg_type == "at":
                qq = data.get("qq")
                if qq:
                    parts.append(f"@{qq}")
            elif self.include_non_text_tokens and seg_type == "image":
                parts.append("[图片]")
            elif self.include_non_text_tokens and seg_type == "face":
                face_id = data.get("id")
                parts.append(f"[表情:{face_id}]")
            elif self.include_non_text_tokens and seg_type in ("video", "record"):
                parts.append("[媒体]")

        return " ".join([p for p in parts if p]).strip()

    def _extract_forward_ids_from_chain(self, chain: list[dict[str, Any]]) -> list[str]:
        ids: list[str] = []
        for seg in chain:
            seg_type = seg.get("type")
            if seg_type not in ("forward", "forward_msg", "nodes"):
                continue
            data = seg.get("data") if isinstance(seg.get("data"), dict) else {}
            fid = data.get("id") or data.get("message_id") or data.get("forward_id")
            if isinstance(fid, (str, int)) and str(fid):
                ids.append(str(fid))
        return ids

    def _extract_sender_from_node(self, node: dict[str, Any]) -> str:
        sender = node.get("sender") if isinstance(node.get("sender"), dict) else {}
        return str(
            sender.get("nickname")
            or sender.get("card")
            or sender.get("user_id")
            or node.get("nickname")
            or node.get("name")
            or ""
        )

    async def _call_get_msg(self, event: AstrMessageEvent, message_id: str) -> dict[str, Any] | None:
        if not message_id.strip():
            return None
        if not hasattr(event, "bot") or not hasattr(event.bot, "api"):
            return None

        mid = message_id.strip()
        params_list: list[dict[str, Any]] = [{"message_id": mid}, {"id": mid}]
        if mid.isdigit():
            params_list.insert(1, {"message_id": int(mid)})
            params_list.append({"id": int(mid)})

        for params in params_list:
            try:
                payload = await event.bot.api.call_action("get_msg", **params)
                if isinstance(payload, dict):
                    return payload
            except Exception:
                continue
        return None

    async def _call_get_forward_msg(
        self,
        event: AstrMessageEvent,
        forward_id: str,
    ) -> dict[str, Any] | None:
        if not forward_id.strip():
            return None
        if not hasattr(event, "bot") or not hasattr(event.bot, "api"):
            return None

        fid = forward_id.strip()
        params_list: list[dict[str, Any]] = [
            {"message_id": fid},
            {"id": fid},
            {"forward_id": fid},
        ]
        if fid.isdigit():
            params_list.insert(1, {"message_id": int(fid)})
            params_list.append({"id": int(fid)})

        for params in params_list:
            try:
                payload = await event.bot.api.call_action("get_forward_msg", **params)
                if isinstance(payload, dict):
                    return payload
            except Exception:
                continue
        return None

    def _extract_sender_name(self, payload: dict[str, Any]) -> str:
        data = self._ob_data(payload)
        sender = data.get("sender") if isinstance(data.get("sender"), dict) else {}
        return str(sender.get("card") or sender.get("nickname") or sender.get("user_id") or "")

    def _extract_text_and_forward_ids(self, payload: dict[str, Any]) -> tuple[str, list[str]]:
        data = self._ob_data(payload)
        message = data.get("message")

        if not isinstance(message, list):
            raw_message = data.get("raw_message")
            if isinstance(raw_message, str) and raw_message.strip():
                return raw_message.strip(), []
            return "", []

        text = self._onebot_chain_to_text(message)
        forward_ids = self._extract_forward_ids_from_chain(message)
        return text, forward_ids

    def _extract_reply_id_from_payload(self, payload: dict[str, Any]) -> str:
        data = self._ob_data(payload)
        message = data.get("message")
        if not isinstance(message, list):
            return ""

        for seg in message:
            if not isinstance(seg, dict):
                continue
            if seg.get("type") != "reply":
                continue
            d = seg.get("data") if isinstance(seg.get("data"), dict) else {}
            rid = d.get("id") or d.get("message_id")
            if isinstance(rid, (str, int)) and str(rid):
                return str(rid)

        return ""

    def _payload_to_reply_segment(self, payload: dict[str, Any] | None) -> Reply | None:
        if not isinstance(payload, dict):
            return None
        data = self._ob_data(payload)
        msg_id = data.get("message_id")
        if not isinstance(msg_id, (str, int)):
            return None

        return Reply(id=str(msg_id), chain=[], message_str="")

    def _get_reply_id(self, reply_seg: Reply) -> str:
        for key in ("id", "message_id", "reply_id", "messageId"):
            val = getattr(reply_seg, key, None)
            if isinstance(val, (str, int)) and str(val):
                return str(val)

        data = getattr(reply_seg, "data", None)
        if isinstance(data, dict):
            for key in ("id", "message_id", "reply"):
                val = data.get(key)
                if isinstance(val, (str, int)) and str(val):
                    return str(val)

        return ""

    def _ob_data(self, payload: Any) -> dict[str, Any]:
        if isinstance(payload, dict):
            data = payload.get("data")
            if isinstance(data, dict):
                return data
            return payload
        return {}

    def _truncate_views(self, views: list[LayerView]) -> list[LayerView]:
        deduped: list[LayerView] = []
        seen: set[str] = set()

        for view in views:
            text = (view.text or "").strip()
            if not text:
                continue
            short = text[: self.max_chars_per_layer]
            key = f"{view.level}|{view.sender}|{short}"
            if key in seen:
                continue
            seen.add(key)
            deduped.append(LayerView(level=view.level, text=short, sender=view.sender))

        total = 0
        capped: list[LayerView] = []
        for view in deduped:
            line_len = len(view.text) + len(view.sender) + 32
            if total + line_len > self.max_total_chars:
                break
            capped.append(view)
            total += line_len

        return capped

    def _format_for_prompt(self, event: AstrMessageEvent, views: list[LayerView]) -> str:
        lines = [
            self.MARKER,
            "[系统注入] 以下是从当前消息引用链/转发链展开得到的多层上下文（按层级排序）。",
            "使用这些信息理解提问语义，优先依据更深层的原始内容，不要编造未出现的细节。",
            "",
        ]

        for view in views:
            sender = f" | sender={view.sender}" if (self.include_sender and view.sender) else ""
            lines.append(f"[Layer {view.level}{sender}] {view.text}")

        gid = ""
        try:
            gid = event.get_group_id() or ""
        except Exception:
            gid = ""
        if gid:
            lines.append(f"\n[group_id] {gid}")

        return "\n".join(lines).strip()
