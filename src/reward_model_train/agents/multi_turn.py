"""Multi-turn generation with tool calling for vLLM and Transformer backends."""

from __future__ import annotations

import json
import logging
from typing import Any

import torch
import torch.distributed
from PIL import Image

from reward_model_train.agents.tool_execution import (
    check_termination_conditions,
    create_tool_response_message,
    parse_tool_call,
    process_tool_result,
    resize_cropped,
)

logger = logging.getLogger(__name__)


DEFAULT_CONFIG = {
    "max_turns": 1,
    "max_images": 16,
    "max_new_tokens": 2048,
    "stop_tokens": ["<|im_end|>", "<|eot_id|>", "<|endoftext|>"],
    "image_sizes": {
        "raw_max": 2000,
        "zoom_max": 1000,
        "select_max": 400,
    },
}


def generate_with_tools_vllm(
    prompts: list[str],
    vllm_engines: list[Any],
    tokenizer: Any,
    data_processor: Any,
    operations: dict[str, Any],
    strategy_args: Any,
    is_eval: bool = False,
    **kwargs,
) -> list[Any]:
    """Multi-turn tool-calling generation using vLLM engines.

    This function orchestrates a multi-turn dialogue loop where the model
    can call tools (frame selection, image cropping) and receive visual
    feedback before generating the final answer.
    """
    import ray
    from vllm import SamplingParams

    config = {
        **DEFAULT_CONFIG,
        "max_turns": getattr(strategy_args, "maxturn", 2) - 1,
        "max_new_tokens": getattr(strategy_args, "max_out_tokens", 2048),
        "temperature": 0.0 if is_eval else getattr(strategy_args, "temperature", 0.85),
        "top_p": 1.0 if is_eval else kwargs.get("top_p", 0.95),
        "image_sizes": {
            **DEFAULT_CONFIG["image_sizes"],
            "eval_min": 256 if is_eval else 4,
            "eval_max": 8000 if is_eval else 5120,
        },
    }

    all_conversations: dict[str, list] = {}
    all_images: dict[str, list] = {}
    all_raw_images: dict[str, list] = {}
    all_outputs: list[Any] = []
    num_tool_calls: list[int] = []
    num_tool_fails: list[int] = []
    video_flags: list[bool] = []

    rank = torch.distributed.get_rank()

    n_samples = 1 if is_eval else strategy_args.n_samples_per_prompt
    expanded_prompts: list[str] = []
    expanded_qids: list[str] = []

    for prompt in prompts:
        info = json.loads(prompt)
        qid = info[-1].get("qid", "unknown")
        clean_prompt = json.dumps(info[:-1])
        for _ in range(n_samples):
            expanded_prompts.append(clean_prompt)
            expanded_qids.append(qid)
            num_tool_calls.append(0)
            num_tool_fails.append(0)

    turn = 0
    active_indices = list(range(len(expanded_prompts)))

    while active_indices and turn <= config["max_turns"]:
        current_vllm_inputs: list[dict] = []
        current_indices: list[int] = []

        for idx in active_indices:
            uid = f"{expanded_qids[idx]}-{idx}"

            if turn == 0:
                message = expanded_prompts[idx]
                prompt_text, conversations = _get_prompt_from_messages(
                    [message], data_processor
                )
                conversations, images, is_video = data_processor.obtain_conv_images_from_conversations(
                    conversations,
                    batch_min_pixels=[config["image_sizes"]["eval_min"] * 28 * 28],
                    batch_max_pixels=[config["image_sizes"]["eval_max"] * 28 * 28],
                )

                all_conversations[uid] = conversations[0]
                all_images[uid] = images[0]
                all_raw_images[uid] = images[0]
                video_flags.append(is_video)

                if is_video and len(images[0]) > 8:
                    step = max(1, len(images[0]) // 8)
                    all_images[uid] = [images[0][::step][:8]]
            else:
                last_output = all_outputs[idx]
                response_text = tokenizer.decode(
                    last_output.outputs[0].token_ids, skip_special_tokens=False
                )

                requires_tool, force_terminate = check_termination_conditions(
                    response_text,
                    num_tool_calls[idx],
                    len(all_images[uid]),
                    len(last_output.prompt_token_ids) + len(last_output.outputs[0].token_ids),
                    config["max_turns"],
                    config["max_images"],
                )

                if not requires_tool or force_terminate:
                    continue

                tool_info = parse_tool_call(response_text)
                if not tool_info:
                    num_tool_fails[idx] += 1
                    continue

                added_images, message_text, error_flag = process_tool_result(
                    tool_info["name"],
                    tool_info["arguments"],
                    all_images[uid],
                    all_raw_images[uid],
                    video_flags[idx],
                    operations,
                    {
                        "select_min_pixels": config["image_sizes"]["eval_min"] * 28 * 28,
                        "select_max_pixels": config["image_sizes"]["select_max"] * 28 * 28,
                        "crop_min_pixels": config["image_sizes"]["eval_min"] * 28 * 28,
                        "crop_max_pixels": config["image_sizes"]["zoom_max"] * 28 * 28,
                    },
                )

                if error_flag:
                    num_tool_fails[idx] += 1

                all_conversations[uid].extend([
                    {"role": "assistant", "content": [{"type": "text", "text": response_text}]},
                    create_tool_response_message(message_text, added_images),
                ])
                all_images[uid].extend(added_images)
                num_tool_calls[idx] += 1

            prompt_text = data_processor.processor.apply_chat_template(
                [all_conversations[uid]], tokenize=False, add_generation_prompt=True
            )[0]

            current_vllm_inputs.append({
                "prompt": prompt_text,
                "multi_modal_data": {
                    ("video" if video_flags[idx] else "image"): all_images[uid]
                },
            })
            current_indices.append(idx)

        if not current_vllm_inputs:
            break

        sampling_params = SamplingParams(
            temperature=config["temperature"] if turn == 0 else 0.9,
            top_p=config["top_p"],
            max_tokens=config["max_new_tokens"],
            stop=config["stop_tokens"],
            include_stop_str_in_output=False,
        )

        batch_size = (len(current_vllm_inputs) + len(vllm_engines) - 1) // len(vllm_engines)
        refs = []
        for i, llm in enumerate(vllm_engines):
            batch = current_vllm_inputs[i * batch_size : (i + 1) * batch_size]
            if batch:
                refs.append(
                    llm.add_requests_vlm.remote(rank, sampling_params=sampling_params, vllm_vision_input=batch)
                )

        ray.get(refs)
        torch.distributed.barrier()

        output_refs = [llm.get_responses.remote(rank) for llm in vllm_engines]
        batch_outputs = sum(ray.get(output_refs), [])

        for i, idx in enumerate(current_indices):
            if turn == 0:
                all_outputs.append(batch_outputs[i])
            else:
                all_outputs[idx] = batch_outputs[i]

        active_indices = current_indices
        turn += 1

    return _build_samples_from_outputs(
        all_outputs, all_conversations, all_images, expanded_qids, tokenizer, data_processor, strategy_args
    )


def generate_with_tools_transformer(
    prompts: list[str],
    model: Any,
    tokenizer: Any,
    data_processor: Any,
    operations: dict[str, Any],
    strategy_args: Any,
    is_eval: bool = False,
    device: str = "cuda",
    **kwargs,
) -> list[Any]:
    """Multi-turn tool-calling generation using standard Transformer model.generate()."""

    config = {
        **DEFAULT_CONFIG,
        "max_turns": getattr(strategy_args, "maxturn", 2) - 1,
        "max_new_tokens": getattr(strategy_args, "max_out_tokens", 2048),
        "temperature": 0.0 if is_eval else getattr(strategy_args, "temperature", 0.85),
        "top_p": 1.0 if is_eval else kwargs.get("top_p", 0.95),
        "image_sizes": {
            **DEFAULT_CONFIG["image_sizes"],
            "eval_min": 256 if is_eval else 4,
            "eval_max": 8000 if is_eval else 5120,
        },
    }

    all_conversations: dict[str, list] = {}
    all_images: dict[str, list] = {}
    all_raw_images: dict[str, list] = {}
    all_outputs: list[Any] = []
    all_generated_texts: list[str] = []
    num_tool_calls: list[int] = []
    num_tool_fails: list[int] = []
    video_flags: list[bool] = []

    n_samples = 1 if is_eval else strategy_args.n_samples_per_prompt
    expanded_prompts: list[str] = []
    expanded_qids: list[str] = []

    for prompt in prompts:
        info = json.loads(prompt)
        qid = info[-1].get("qid", "unknown")
        clean_prompt = json.dumps(info[:-1])
        for _ in range(n_samples):
            expanded_prompts.append(clean_prompt)
            expanded_qids.append(qid)
            num_tool_calls.append(0)
            num_tool_fails.append(0)

    turn = 0
    active_indices = list(range(len(expanded_prompts)))

    while active_indices and turn <= config["max_turns"]:
        batch_inputs: list[torch.Tensor] = []
        batch_visual_inputs: list[dict] = []
        current_indices: list[int] = []

        for idx in active_indices:
            uid = f"{expanded_qids[idx]}-{idx}"

            if turn == 0:
                conversations = json.loads(expanded_prompts[idx])
                processed_conv, images, is_video = data_processor.obtain_conv_images_from_conversations(
                    [conversations],
                    batch_min_pixels=[config["image_sizes"]["eval_min"] * 28 * 28],
                    batch_max_pixels=[config["image_sizes"]["eval_max"] * 28 * 28],
                )

                all_conversations[uid] = processed_conv[0]
                all_images[uid] = images[0]
                all_raw_images[uid] = images[0]
                video_flags.append(is_video)

                if is_video and len(images[0]) > 8:
                    step = max(1, len(images[0]) // 8)
                    all_images[uid] = [images[0][::step][:8]]
            else:
                last_response = all_generated_texts[idx]
                requires_tool, force_terminate = check_termination_conditions(
                    last_response,
                    num_tool_calls[idx],
                    len(all_images[uid]),
                    len(tokenizer.encode(last_response)),
                    config["max_turns"],
                    config["max_images"],
                )

                if not requires_tool or force_terminate:
                    continue

                tool_info = parse_tool_call(last_response)
                if not tool_info:
                    num_tool_fails[idx] += 1
                    continue

                added_images, message_text, error_flag = process_tool_result(
                    tool_info["name"],
                    tool_info["arguments"],
                    all_images[uid],
                    all_raw_images[uid],
                    video_flags[idx],
                    operations,
                    {
                        "select_min_pixels": config["image_sizes"]["eval_min"] * 28 * 28,
                        "select_max_pixels": config["image_sizes"]["select_max"] * 28 * 28,
                        "crop_min_pixels": config["image_sizes"]["eval_min"] * 28 * 28,
                        "crop_max_pixels": config["image_sizes"]["zoom_max"] * 28 * 28,
                    },
                )

                if error_flag:
                    num_tool_fails[idx] += 1

                all_conversations[uid].extend([
                    {"role": "assistant", "content": [{"type": "text", "text": last_response}]},
                    create_tool_response_message(message_text, added_images),
                ])
                all_images[uid].extend(added_images)
                num_tool_calls[idx] += 1

            text_input = data_processor.processor.apply_chat_template(
                [all_conversations[uid]], tokenize=False, add_generation_prompt=True
            )[0]

            if video_flags[idx]:
                visual_inputs = data_processor.processor(
                    text=[text_input],
                    videos=[all_images[uid][0]] if isinstance(all_images[uid][0], list) else [all_images[uid]],
                    images=all_images[uid][1:] if len(all_images[uid]) > 1 else None,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=20000,
                )
            else:
                visual_inputs = data_processor.processor(
                    text=[text_input],
                    images=all_images[uid],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=20000,
                )

            batch_inputs.append(visual_inputs["input_ids"])
            batch_visual_inputs.append(
                {k: v for k, v in visual_inputs.items() if k not in ("input_ids", "attention_mask")}
            )
            current_indices.append(idx)

        if not batch_inputs:
            break

        input_ids = torch.nn.utils.rnn.pad_sequence(
            batch_inputs, batch_first=True, padding_value=tokenizer.pad_token_id
        ).to(device)
        attention_mask = input_ids.ne(tokenizer.pad_token_id).to(device)

        combined_visual = {}
        for key in batch_visual_inputs[0]:
            if all(key in vi for vi in batch_visual_inputs):
                combined_visual[key] = torch.cat([vi[key] for vi in batch_visual_inputs], dim=0).to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **combined_visual,
                max_new_tokens=config["max_new_tokens"],
                temperature=config["temperature"] if turn == 0 else 0.9,
                top_p=config["top_p"],
                do_sample=not is_eval,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_ids = outputs[:, input_ids.shape[1] :]
        generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)

        for i, idx in enumerate(current_indices):
            if turn == 0:
                all_outputs.append(outputs[i])
                all_generated_texts.append(generated_texts[i])
            else:
                all_outputs[idx] = outputs[i]
                all_generated_texts[idx] = generated_texts[i]

        active_indices = current_indices
        turn += 1

    return _build_samples_from_transformer_outputs(
        all_outputs, all_generated_texts, all_conversations, all_images,
        expanded_qids, tokenizer, data_processor, strategy_args,
    )


def _get_prompt_from_messages(messages, data_processor):
    """Extract prompt text from messages (placeholder for actual implementation)."""
    # This hooks into the data_processor's chat template
    pass


def _build_samples_from_outputs(outputs, conversations, images, qids, tokenizer, data_processor, args):
    """Build Sample objects from vLLM outputs (placeholder for actual implementation)."""
    pass


def _build_samples_from_transformer_outputs(
    outputs, generated_texts, conversations, images, qids, tokenizer, data_processor, args
):
    """Build Sample objects from Transformer outputs (placeholder for actual implementation)."""
    pass
