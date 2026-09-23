"""Config classes for checkpoints whose `model_type` is newer than the installed transformers.

A serving engine can know a model that `transformers` does not. vLLM 0.17.0 ships
`glm4_moe_lite.py`, so it can build GLM-4.7-Flash, but reading the checkpoint's config goes through
`AutoConfig`, and transformers 4.57.6 answers:

    ValueError: The checkpoint you are trying to load has model type `glm4_moe_lite` but
    Transformers does not recognize this architecture.

There is no 4.x release that knows the type -- the checkpoint was saved by 5.0.0rc0, and the version
list goes 4.57.6 -> 5.0.0 -- while vLLM 0.17.0 requires `transformers<5`. Upgrading is therefore not an
option, and it would in any case put a major version bump under the multimodal suite that shares this
environment.

What is actually missing is small. `PretrainedConfig.__init__` keeps unrecognised keyword arguments as
attributes, so a subclass carrying nothing but the right `model_type` gives vLLM a config object with
every field the checkpoint declares. The engine reads those fields directly; nothing here has to know
what they mean.

Two constraints shaped the form of this:

- The classes live in a module rather than in a script, because vLLM pickles the config to its engine
  subprocess. A class defined in `__main__` cannot be unpickled there.
- Registration is skipped when transformers already knows the type, so this stops being load-bearing on
  its own the moment the environment is upgraded, instead of shadowing the real class.
"""
from transformers import AutoConfig, PretrainedConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING


class Glm4MoeLiteConfig(PretrainedConfig):
    """GLM-4.7-Flash: MoE with MLA attention, sigmoid-scored router, one shared expert."""

    model_type = "glm4_moe_lite"


SHIMS: tuple[type[PretrainedConfig], ...] = (Glm4MoeLiteConfig,)


def register_missing_configs() -> list[str]:
    """Register shims for model types this transformers does not have. Returns what was added.

    Membership has to be tested against `CONFIG_MAPPING`, not the built-in `CONFIG_MAPPING_NAMES`:
    `AutoConfig.register` writes into the mapping's extra content and only raises for names that are
    built in, so a check against the built-in table would report a fresh registration on every call.
    """
    added = []
    for cls in SHIMS:
        if cls.model_type in CONFIG_MAPPING:
            continue
        AutoConfig.register(cls.model_type, cls)
        added.append(cls.model_type)
    return added


# 各家 checkpoint 对「每个 token 激活几个专家」这个字段的叫法并不统一：Qwen3 放在
# 顶层 num_experts_per_tok，Qwen3.5 放在 text_config 下，Gemma 4 叫 top_k_experts。
#
# 关键在于：config 里用的是哪个名字，vLLM 读的就是哪个名字。所以 overlay 必须写回
# 同一个键，而不能新增一个自己偏好的键——给一个用 top_k_experts 的 config 补上
# num_experts_per_tok，vLLM 会直接忽略它，于是这次运行会以未剪枝的预算计算，却在
# 报告里标成剪枝后的结果。那是最坏的一种失败：安静，且数字看起来是对的。
#
# 顺序即优先级，新键只作为兜底，以保持已发表运行的行为逐位不变。
_MOE_TOPK_SITES = (
    ((), "num_experts_per_tok"),               # Qwen3, DeepSeek, MiniMax, Ling
    (("text_config",), "num_experts_per_tok"),  # Qwen3.5
    ((), "top_k_experts"),
    (("text_config",), "top_k_experts"),        # Gemma 4
)


def moe_topk_site(config: dict) -> tuple[dict, str] | None:
    """定位持有「每 token 专家数」的 (容器, 键)；不是 MoE config 时返回 None。"""
    for path, key in _MOE_TOPK_SITES:
        container = config
        for part in path:
            container = container.get(part) if isinstance(container, dict) else None
            if container is None:
                break
        if isinstance(container, dict) and key in container:
            return container, key
    return None


def get_moe_num_experts_per_tok(config: dict) -> int | None:
    """读取每 token 专家数，兼容各家 checkpoint 的不同字段名与嵌套位置。"""
    site = moe_topk_site(config)
    return site[0][site[1]] if site is not None else None


def set_moe_num_experts_per_tok(config: dict, num_experts_per_tok: int) -> None:
    """就地改写每 token 专家数，写回 config 本来使用的那个键。"""
    site = moe_topk_site(config)
    if site is None:
        raise ValueError(
            "config.json 中找不到每 token 专家数字段（顶层或 text_config 下的 "
            f"{' / '.join(sorted({k for _, k in _MOE_TOPK_SITES}))}），"
            "当前可能不是 MoE 模型")
    container, key = site
    container[key] = num_experts_per_tok
    # gpt-oss keeps a duplicate field; vLLM may read either.
    if "experts_per_token" in config:
        config["experts_per_token"] = num_experts_per_tok
