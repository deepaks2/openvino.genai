// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/bagel/pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <numeric>
#include <random>
#include <set>
#include <sstream>

#include "openvino/core/type/float16.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/runtime/core.hpp"
#include "utils.hpp"
#include "visual_language/clip.hpp"

namespace ov::genai {
namespace {

using Clock = std::chrono::steady_clock;

float ms_since(Clock::time_point t0) {
    return std::chrono::duration<float, std::milli>(Clock::now() - t0).count();
}

std::vector<size_t> parse_buckets(const std::string& text) {
    std::vector<size_t> buckets;
    std::stringstream ss(text);
    for (std::string item; std::getline(ss, item, ',');) {
        buckets.push_back(std::stoul(item));
    }
    return buckets;
}

size_t pick_bucket(const std::vector<size_t>& buckets, size_t n) {
    const auto it = std::find_if(buckets.begin(), buckets.end(), [n](size_t b) {
        return b >= n;
    });
    return it == buckets.end() ? n : *it;
}

// numpy.random.RandomState(seed).randn: MT19937 with the legacy polar Box-Muller transform
class NumpyRandomState {
public:
    explicit NumpyRandomState(uint32_t seed) : m_mt(seed) {}

    double gauss() {
        if (m_has_gauss) {
            m_has_gauss = false;
            return m_gauss;
        }
        double x1, x2, r2;
        do {
            x1 = 2.0 * next_double() - 1.0;
            x2 = 2.0 * next_double() - 1.0;
            r2 = x1 * x1 + x2 * x2;
        } while (r2 >= 1.0 || r2 == 0.0);
        const double f = std::sqrt(-2.0 * std::log(r2) / r2);
        m_gauss = f * x1;
        m_has_gauss = true;
        return f * x2;
    }

private:
    double next_double() {
        const uint32_t a = m_mt() >> 5;
        const uint32_t b = m_mt() >> 6;
        return (a * 67108864.0 + b) / 9007199254740992.0;
    }

    std::mt19937 m_mt;
    bool m_has_gauss = false;
    double m_gauss = 0.0;
};

// Python round(): half to even
int py_round(double v) {
    return static_cast<int>(std::nearbyint(v));
}

// BAGEL ImageTransform: MaxLongEdgeMinShortEdgeResize (Pillow bicubic), ToTensor, Normalize(0.5, 0.5)
struct ImageTransform {
    int max_size;
    int min_size;
    int stride;
    double max_pixels = 14.0 * 14 * 9 * 1024;

    int make_divisible(int v) const {
        return std::max(stride, py_round(static_cast<double>(v) / stride) * stride);
    }

    std::pair<int, int> apply_scale(int w, int h, double scale) const {
        return {make_divisible(py_round(w * scale)), make_divisible(py_round(h * scale))};
    }

    // Returns a [3, H, W] f32 tensor
    ov::Tensor operator()(const ov::Tensor& image) const {
        const auto& shape = image.get_shape();
        OPENVINO_ASSERT(image.get_element_type() == ov::element::u8 &&
                            (shape.size() == 3 || (shape.size() == 4 && shape[0] == 1)) && shape.back() == 3,
                        "BAGEL expects a u8 RGB image with [1, H, W, 3] or [H, W, 3] layout");
        const int h = static_cast<int>(shape[shape.size() - 3]);
        const int w = static_cast<int>(shape[shape.size() - 2]);
        OPENVINO_ASSERT(h > 0 && w > 0, "BAGEL expects a non-empty image");

        double scale = std::min(static_cast<double>(max_size) / std::max(w, h), 1.0);
        scale = std::max(scale, static_cast<double>(min_size) / std::min(w, h));
        auto [nw, nh] = apply_scale(w, h, scale);
        if (static_cast<double>(nw) * nh > max_pixels) {
            std::tie(nw, nh) = apply_scale(nw, nh, max_pixels / (static_cast<double>(nw) * nh));
        }
        if (std::max(nw, nh) > max_size) {
            std::tie(nw, nh) = apply_scale(nw, nh, static_cast<double>(max_size) / std::max(nw, nh));
        }

        clip_image_u8 src{w, h, {image.data<uint8_t>(), image.data<uint8_t>() + image.get_size()}};
        clip_image_u8 dst;
        bicubic_resize(src, dst, nw, nh);

        ov::Tensor out(ov::element::f32, {3, static_cast<size_t>(nh), static_cast<size_t>(nw)});
        float* o = out.data<float>();
        const size_t plane = static_cast<size_t>(nh) * nw;
        for (size_t i = 0; i < plane; ++i) {
            for (size_t c = 0; c < 3; ++c) {
                o[c * plane + i] = (static_cast<float>(dst.buf[i * 3 + c]) / 255.0f - 0.5f) / 0.5f;
            }
        }
        return out;
    }
};

// [C, H, W] -> [(H / p) * (W / p), p * p * C] in (row, col, py, px, c) order; crops to the given grid
ov::Tensor patchify(const float* src, size_t c, size_t h, size_t w, size_t p, size_t grid_h, size_t grid_w) {
    ov::Tensor out(ov::element::f32, {grid_h * grid_w, p * p * c});
    float* o = out.data<float>();
    for (size_t gh = 0; gh < grid_h; ++gh) {
        for (size_t gw = 0; gw < grid_w; ++gw) {
            for (size_t py = 0; py < p; ++py) {
                for (size_t px = 0; px < p; ++px) {
                    for (size_t ch = 0; ch < c; ++ch) {
                        *o++ = src[(ch * h + gh * p + py) * w + gw * p + px];
                    }
                }
            }
        }
    }
    return out;
}

std::vector<int64_t> flat_position_ids(size_t grid_h, size_t grid_w, size_t max_per_side) {
    std::vector<int64_t> ids;
    ids.reserve(grid_h * grid_w);
    for (size_t r = 0; r < grid_h; ++r) {
        for (size_t c = 0; c < grid_w; ++c) {
            ids.push_back(static_cast<int64_t>(r * max_per_side + c));
        }
    }
    return ids;
}

ov::Tensor i64_tensor(const std::vector<int64_t>& values) {
    ov::Tensor t(ov::element::i64, {values.size()});
    std::copy(values.begin(), values.end(), t.data<int64_t>());
    return t;
}

ov::Tensor i64_batch_tensor(const std::vector<int64_t>& values) {
    ov::Tensor t(ov::element::i64, {1, values.size()});
    std::copy(values.begin(), values.end(), t.data<int64_t>());
    return t;
}

std::vector<int64_t> iota_i64(int64_t begin, int64_t end) {
    std::vector<int64_t> v(static_cast<size_t>(std::max<int64_t>(end - begin, 0)));
    std::iota(v.begin(), v.end(), begin);
    return v;
}

ov::Tensor copy_tensor(const ov::Tensor& src) {
    ov::Tensor dst(src.get_element_type(), src.get_shape());
    if (src.get_size() > 0) {
        std::memcpy(dst.data(), src.data(), src.get_byte_size());
    }
    return dst;
}

ov::Tensor causal_mask(size_t q, size_t past) {
    ov::Tensor m(ov::element::f32, {1, 1, q, past + q});
    float* d = m.data<float>();
    std::fill_n(d, m.get_size(), 0.0f);
    for (size_t r = 0; r < q; ++r) {
        std::fill(d + r * (past + q) + past + r + 1, d + (r + 1) * (past + q), -std::numeric_limits<float>::infinity());
    }
    return m;
}

ov::Tensor full_mask(size_t q, size_t past) {
    ov::Tensor m(ov::element::f32, {1, 1, q, past + q});
    std::fill_n(m.data<float>(), m.get_size(), 0.0f);
    return m;
}

float l2_norm(const std::vector<float>& v) {
    double s = 0.0;
    for (float x : v) {
        s += static_cast<double>(x) * x;
    }
    return static_cast<float>(std::sqrt(s));
}

ov::Tensor tensor_from_f32(const ov::element::Type& type, const ov::Shape& shape, const std::vector<float>& values) {
    OPENVINO_ASSERT(values.size() == ov::shape_size(shape),
                    "Tensor value count does not match shape: ",
                    values.size(),
                    " versus ",
                    ov::shape_size(shape));
    ov::Tensor tensor(type, shape);
    if (type == ov::element::f32) {
        std::copy(values.begin(), values.end(), tensor.data<float>());
    } else if (type == ov::element::f16) {
        std::transform(values.begin(), values.end(), tensor.data<ov::float16>(), [](float value) {
            return static_cast<ov::float16>(value);
        });
    } else if (type == ov::element::bf16) {
        std::transform(values.begin(), values.end(), tensor.data<ov::bfloat16>(), [](float value) {
            return static_cast<ov::bfloat16>(value);
        });
    } else {
        OPENVINO_ASSERT(false, "Unsupported floating-point tensor element type: ", type);
    }
    return tensor;
}

std::vector<float> tensor_to_f32(const ov::Tensor& tensor) {
    std::vector<float> values(tensor.get_size());
    if (tensor.get_element_type() == ov::element::f32) {
        std::copy(tensor.data<const float>(), tensor.data<const float>() + tensor.get_size(), values.begin());
    } else if (tensor.get_element_type() == ov::element::f16) {
        std::transform(tensor.data<const ov::float16>(),
                       tensor.data<const ov::float16>() + tensor.get_size(),
                       values.begin(),
                       [](ov::float16 value) {
                           return static_cast<float>(value);
                       });
    } else if (tensor.get_element_type() == ov::element::bf16) {
        std::transform(tensor.data<const ov::bfloat16>(),
                       tensor.data<const ov::bfloat16>() + tensor.get_size(),
                       values.begin(),
                       [](ov::bfloat16 value) {
                           return static_cast<float>(value);
                       });
    } else {
        OPENVINO_ASSERT(false, "Unsupported floating-point tensor element type: ", tensor.get_element_type());
    }
    return values;
}

}  // namespace

class BagelPipeline::Impl {
public:
    Impl(const std::filesystem::path& models_dir,
         const std::string& device,
         const std::filesystem::path& tokenizer_dir,
         const ov::AnyMap& properties)
        : m_npu(device.rfind("NPU", 0) == 0),
          m_tokenizer(tokenizer_dir.empty() ? models_dir : tokenizer_dir) {
        if (std::filesystem::is_regular_file(models_dir / "bagel_openvino_config.json")) {
            init_static_export(models_dir, device, properties);
            return;
        }
        std::ifstream f(models_dir / "bagel_ov_config.json");
        OPENVINO_ASSERT(f.is_open(), "Failed to open ", (models_dir / "bagel_ov_config.json").string());
        const auto cfg = nlohmann::json::parse(f);
        m_hidden = cfg.at("hidden_size").get<size_t>();
        m_layers = cfg.at("num_layers").get<size_t>();
        m_kv_heads = cfg.at("num_kv_heads").get<size_t>();
        m_head_dim = cfg.at("head_dim").get<size_t>();
        m_latent_channel = cfg.at("latent_channel").get<size_t>();
        m_latent_patch = cfg.at("latent_patch_size").get<size_t>();
        m_latent_downsample = cfg.at("latent_downsample").get<size_t>();
        m_vit_patch = cfg.at("vit_patch_size").get<size_t>();
        m_vit_max_side = cfg.at("vit_max_num_patch_per_side").get<size_t>();
        m_max_latent = cfg.at("max_latent_size").get<size_t>();
        const auto& ids = cfg.at("new_token_ids");
        m_bos = ids.at("bos_token_id").get<int64_t>();
        m_eos = ids.at("eos_token_id").get<int64_t>();
        m_soi = ids.at("start_of_image").get<int64_t>();
        m_eoi = ids.at("end_of_image").get<int64_t>();

        // On NPU the dynamic IRs go through NPUW_BAGEL, which compiles static variants on first use;
        // the text embedding lookup stays on CPU like in GenAI VLMs
        ov::AnyMap npu_config{{"NPU_USE_NPUW", "YES"}, {"NPUW_BAGEL", "YES"}};
        for (const auto& [k, v] : properties) {
            npu_config[k] = v;
        }

        auto& core = utils::singleton_core();
        if (m_npu) {
            // Only used to name the NPUW_BAGEL static variant each call compiles
            const auto buckets = [&](const std::string& key) {
                const auto it = properties.find(key);
                return parse_buckets(it != properties.end() ? it->second.as<std::string>()
                                                            : core.get_property(device, key).as<std::string>());
            };
            m_seq_buckets = buckets("NPUW_BAGEL_SEQ_BUCKETS");
            m_kv_buckets = buckets("NPUW_BAGEL_KV_BUCKETS");
        }
        const auto compile = [&](const std::string& file, const std::string& dev, const ov::AnyMap& config) {
            const auto t0 = Clock::now();
            auto compiled = core.compile_model((models_dir / file).string(), dev, config);
            m_perf.compile_ms.emplace_back("compile_model " + file.substr(0, file.find("_model.xml")), ms_since(t0));
            return compiled.create_infer_request();
        };
        const auto& config = m_npu ? npu_config : properties;
        m_text_emb = m_npu ? compile("text_embeddings_model.xml", "CPU", {})
                           : compile("text_embeddings_model.xml", device, properties);
        m_vit = compile("vision_embeddings_model.xml", device, config);
        m_lm = compile("language_model.xml", device, config);
        m_vae_enc = compile("vae_encoder_model.xml", device, config);
        m_vae_dec = compile("vae_decoder_model.xml", device, config);
    }

    std::string generate_text(const std::string& prompt,
                              const std::optional<ov::Tensor>& image,
                              size_t max_new_tokens,
                              const std::string& system_prompt) {
        if (m_static_export) {
            return generate_text_static(prompt, image, max_new_tokens, system_prompt);
        }
        reset();
        if (!system_prompt.empty()) {
            prefill_text(system_prompt);
        }
        if (image) {
            prefill_image(*image);
        }
        prefill_text(prompt);

        int64_t cur = m_bos;
        std::vector<int64_t> out;
        for (size_t i = 0; i < max_new_tokens; ++i) {
            const auto logits = lm(embed({cur}), {m_ctx.rope}, full_mask(1, m_ctx.len), true, "decode");
            ++m_ctx.len;
            ++m_ctx.rope;
            const size_t vocab = logits.get_shape().back();
            const float* last = logits.data<const float>() + logits.get_size() - vocab;
            cur = std::max_element(last, last + vocab) - last;
            if (cur == m_eos) {
                break;
            }
            out.push_back(cur);
        }
        return m_tokenizer.decode(out, {{"skip_special_tokens", false}});
    }

    ov::Tensor generate_image(const std::string& prompt, const BagelImageGenerationConfig& config) {
        if (m_static_export) {
            return generate_image_static(prompt, config);
        }
        reset();
        prefill_text(prompt);
        const Context main = m_ctx;
        std::optional<Context> cfg_text;
        if (config.cfg_text_scale > 1.0f) {
            cfg_text = Context{empty_kv(), 0, 0};
        }
        const size_t side = config.size / m_latent_downsample;
        OPENVINO_ASSERT(side > 0, "size must be at least ", m_latent_downsample);
        return diffuse(side, side, main, cfg_text, std::nullopt, config, {0.4f, 1.0f});
    }

    ov::Tensor edit_image(const ov::Tensor& image,
                          const std::string& prompt,
                          const BagelImageGenerationConfig& config) {
        if (m_static_export) {
            return edit_image_static(image, prompt, config);
        }
        const auto [vae_ctx, hw] = vae_prefill(image, Context{empty_kv(), 0, 0});

        reset();
        m_ctx = vae_ctx;
        prefill_image(image);
        const Context img_ctx = m_ctx;
        prefill_text(prompt);
        const Context main = m_ctx;

        std::optional<Context> cfg_text, cfg_img;
        if (config.cfg_text_scale > 1.0f) {
            cfg_text = img_ctx;
        }
        if (config.cfg_img_scale > 1.0f) {
            reset();
            prefill_text(prompt);
            cfg_img = m_ctx;
        }
        return diffuse(hw.first, hw.second, main, cfg_text, cfg_img, config, {0.0f, 1.0f});
    }

    BagelPerfMetrics m_perf;

private:
    struct StaticContext {
        std::vector<float> embeds;
        std::vector<int64_t> position_ids;
        std::vector<uint8_t> generation_mask;
        std::vector<float> attention_mask;
        size_t len = 0;
        int64_t rope = 0;
    };

    struct StaticOutputs {
        std::vector<float> logits;
        std::vector<float> velocity;
    };

    int64_t static_token_id(const std::string& token) {
        const auto ids = m_tokenizer.encode(token).input_ids;
        OPENVINO_ASSERT(ids.get_size() == 1,
                        "BAGEL special token ",
                        token,
                        " must encode to one token, got ",
                        ids.get_size());
        return ids.data<const int64_t>()[0];
    }

    void init_static_export(const std::filesystem::path& models_dir,
                            const std::string& device,
                            const ov::AnyMap& properties) {
        m_static_export = true;
        std::ifstream metadata_file(models_dir / "bagel_openvino_config.json");
        OPENVINO_ASSERT(metadata_file.is_open(),
                        "Failed to open ",
                        (models_dir / "bagel_openvino_config.json").string());
        const auto metadata = nlohmann::json::parse(metadata_file);

        std::ifstream model_config_file(models_dir / "config.json");
        OPENVINO_ASSERT(model_config_file.is_open(), "Failed to open ", (models_dir / "config.json").string());
        const auto model_config = nlohmann::json::parse(model_config_file);

        m_static_sequence_length = metadata.at("sequence_length").get<size_t>();
        m_static_vision_tokens = metadata.at("vision_tokens").get<size_t>();
        m_hidden = metadata.at("hidden_size").get<size_t>();
        m_latent_channel = metadata.at("latent_channels").get<size_t>();
        m_latent_patch = metadata.at("latent_patch_size").get<size_t>();
        m_latent_downsample = metadata.at("latent_downsample").get<size_t>();
        m_max_latent = model_config.at("max_latent_size").get<size_t>();
        m_vit_patch = model_config.at("vit_config").at("patch_size").get<size_t>();
        m_vit_max_side = model_config.at("vit_max_num_patch_per_side").get<size_t>();
        m_static_max_latent_tokens = metadata.at("max_latent_tokens").get<size_t>();
        m_static_export_hidden = m_hidden;
        m_static_patch_latent_dim = m_latent_patch * m_latent_patch * m_latent_channel;

        const auto& llm_config = model_config.at("llm_config");
        m_bos = llm_config.at("bos_token_id").get<int64_t>();
        m_eos = llm_config.at("eos_token_id").get<int64_t>();
        m_soi = static_token_id("<|vision_start|>");
        m_eoi = static_token_id("<|vision_end|>");

        OPENVINO_ASSERT(m_static_sequence_length > 0, "BAGEL sequence_length must be positive");
        OPENVINO_ASSERT(m_static_vision_tokens > 0, "BAGEL vision_tokens must be positive");
        OPENVINO_ASSERT(m_hidden > 0, "BAGEL hidden_size must be positive");
        OPENVINO_ASSERT(m_latent_channel > 0, "BAGEL latent_channels must be positive");
        OPENVINO_ASSERT(m_latent_patch > 0, "BAGEL latent_patch_size must be positive");
        OPENVINO_ASSERT(m_latent_downsample > 0, "BAGEL latent_downsample must be positive");
        OPENVINO_ASSERT(m_static_max_latent_tokens > 0, "BAGEL max_latent_tokens must be positive");
        OPENVINO_ASSERT(m_vit_patch > 0, "BAGEL vit patch size must be positive");
        OPENVINO_ASSERT(m_vit_max_side > 0, "BAGEL vit_max_num_patch_per_side must be positive");

        auto& core = utils::singleton_core();
        const auto compile = [&](const std::string& file, const std::string& compile_device) {
            const auto t0 = Clock::now();
            auto compiled = core.compile_model((models_dir / file).string(), compile_device, properties);
            m_perf.compile_ms.emplace_back("compile_model " + file.substr(0, file.find("_model.xml")), ms_since(t0));
            return compiled.create_infer_request();
        };
        const std::string text_device = m_npu ? "CPU" : device;
        m_static_text_emb = compile("openvino_text_embeddings_model.xml", text_device);
        m_static_vit = compile("openvino_vision_model.xml", device);
        m_static_latent_emb = compile("openvino_latent_embeddings_model.xml", device);
        m_static_lm = compile("openvino_language_model.xml", device);
        m_static_vae_enc = compile("openvino_vae_encoder_model.xml", device);
        m_static_vae_dec = compile("openvino_vae_decoder_model.xml", device);

        const auto latent_shape = m_static_vae_dec.get_tensor("latents").get_shape();
        const auto image_shape = m_static_vae_dec.get_output_tensor().get_shape();
        OPENVINO_ASSERT(latent_shape.size() == 4 && image_shape.size() == 4,
                        "BAGEL VAE graphs must use four-dimensional tensors");
        OPENVINO_ASSERT(latent_shape[0] == 1 && image_shape[0] == 1 && image_shape[1] == 3,
                        "BAGEL VAE graphs must use batch one and three output channels");
        m_static_latent_height = latent_shape[2];
        m_static_latent_width = latent_shape[3];
        m_static_image_height = image_shape[2];
        m_static_image_width = image_shape[3];
        OPENVINO_ASSERT(m_static_latent_height % m_latent_patch == 0 &&
                            m_static_latent_width % m_latent_patch == 0,
                        "BAGEL VAE latent shape must be divisible by latent_patch_size");
    }

    std::vector<float> static_text_embeddings(const std::vector<int64_t>& ids) {
        OPENVINO_ASSERT(ids.size() <= m_static_sequence_length,
                        "BAGEL text sequence exceeds exported sequence length: ",
                        ids.size(),
                        " > ",
                        m_static_sequence_length);
        std::vector<int64_t> padded(m_static_sequence_length);
        std::copy(ids.begin(), ids.end(), padded.begin());
        ov::Tensor input(ov::element::i64, {1, m_static_sequence_length});
        std::copy(padded.begin(), padded.end(), input.data<int64_t>());
        const auto t0 = Clock::now();
        m_static_text_emb.set_tensor("input_ids", input);
        m_static_text_emb.infer();
        const auto values = tensor_to_f32(m_static_text_emb.get_output_tensor());
        record("text_emb", t0);
        return {values.begin(), values.begin() + ids.size() * m_static_export_hidden};
    }

    void static_append(StaticContext& context,
                       const std::vector<float>& embeds,
                       const std::vector<int64_t>& position_ids,
                       const std::vector<uint8_t>& generation_mask,
                       bool full_attention) {
        OPENVINO_ASSERT(embeds.size() % m_static_export_hidden == 0,
                        "BAGEL embedding sequence has an invalid hidden dimension");
        const size_t count = embeds.size() / m_static_export_hidden;
        OPENVINO_ASSERT(position_ids.size() == count && generation_mask.size() == count,
                        "BAGEL context metadata does not match embedding sequence");
        const size_t old_len = context.len;
        const size_t new_len = old_len + count;
        OPENVINO_ASSERT(new_len <= m_static_sequence_length,
                        "BAGEL context exceeds exported sequence length: ",
                        new_len,
                        " > ",
                        m_static_sequence_length);

        std::vector<float> new_embeds(new_len * m_static_export_hidden);
        std::copy(context.embeds.begin(), context.embeds.end(), new_embeds.begin());
        std::copy(embeds.begin(), embeds.end(), new_embeds.begin() + old_len * m_static_export_hidden);
        context.embeds = std::move(new_embeds);

        context.position_ids.insert(context.position_ids.end(), position_ids.begin(), position_ids.end());
        context.generation_mask.insert(context.generation_mask.end(), generation_mask.begin(), generation_mask.end());

        std::vector<float> new_mask(new_len * new_len, -std::numeric_limits<float>::infinity());
        for (size_t row = 0; row < old_len; ++row) {
            std::copy(context.attention_mask.begin() + row * old_len,
                      context.attention_mask.begin() + (row + 1) * old_len,
                      new_mask.begin() + row * new_len);
        }
        for (size_t row = old_len; row < new_len; ++row) {
            for (size_t col = 0; col < old_len; ++col) {
                new_mask[row * new_len + col] = 0.0f;
            }
            const size_t local_row = row - old_len;
            for (size_t col = 0; col < count; ++col) {
                if (full_attention || col <= local_row) {
                    new_mask[row * new_len + old_len + col] = 0.0f;
                }
            }
        }
        context.attention_mask = std::move(new_mask);
        context.len = new_len;
    }

    void static_append_text(StaticContext& context, const std::vector<int64_t>& ids) {
        const auto embeds = static_text_embeddings(ids);
        std::vector<int64_t> positions(ids.size());
        std::iota(positions.begin(), positions.end(), context.rope);
        static_append(context, embeds, positions, std::vector<uint8_t>(ids.size()), false);
        context.rope += static_cast<int64_t>(ids.size());
    }

    StaticContext static_text_context(const std::string& text) {
        StaticContext context;
        static_append_text(context, {m_bos});
        const auto tokenized = m_tokenizer.encode(text).input_ids;
        std::vector<int64_t> ids(tokenized.data<const int64_t>(),
                                 tokenized.data<const int64_t>() + tokenized.get_size());
        static_append_text(context, ids);
        static_append_text(context, {m_eos});
        return context;
    }

    void static_add_text(StaticContext& context, const std::string& text) {
        static_append_text(context, {m_bos});
        const auto tokenized = m_tokenizer.encode(text).input_ids;
        std::vector<int64_t> ids(tokenized.data<const int64_t>(),
                                 tokenized.data<const int64_t>() + tokenized.get_size());
        static_append_text(context, ids);
        static_append_text(context, {m_eos});
    }

    void static_append_understanding(StaticContext& context, const std::vector<float>& image_embeds) {
        const size_t count = image_embeds.size() / m_static_export_hidden;
        std::vector<float> embeds;
        embeds.reserve((count + 2) * m_static_export_hidden);
        const auto start = static_text_embeddings({m_soi});
        const auto end = static_text_embeddings({m_eoi});
        embeds.insert(embeds.end(), start.begin(), start.end());
        embeds.insert(embeds.end(), image_embeds.begin(), image_embeds.end());
        embeds.insert(embeds.end(), end.begin(), end.end());
        static_append(context,
                      embeds,
                      std::vector<int64_t>(count + 2, context.rope),
                      std::vector<uint8_t>(count + 2),
                      true);
        ++context.rope;
    }

    void static_append_generation(StaticContext& context, const std::vector<float>& latent_embeds) {
        const size_t count = latent_embeds.size() / m_static_export_hidden;
        OPENVINO_ASSERT(count > 0, "BAGEL generation sequence must contain at least one latent token");
        std::vector<float> embeds;
        embeds.reserve((count + 2) * m_static_export_hidden);
        const auto start = static_text_embeddings({m_soi});
        const auto end = static_text_embeddings({m_eoi});
        embeds.insert(embeds.end(), start.begin(), start.end());
        embeds.insert(embeds.end(), latent_embeds.begin(), latent_embeds.end());
        embeds.insert(embeds.end(), end.begin(), end.end());
        std::vector<uint8_t> generation(count + 2);
        if (generation.size() > 2) {
            std::fill_n(generation.begin() + 1, generation.size() - 2, uint8_t{1});
        }
        static_append(context,
                      embeds,
                      std::vector<int64_t>(count + 2, context.rope),
                      generation,
                      true);
        ++context.rope;
    }

    StaticOutputs static_forward(const StaticContext& context) {
        OPENVINO_ASSERT(context.len <= m_static_sequence_length, "BAGEL context exceeds exported sequence length");
        const ov::Shape sequence_shape{1, m_static_sequence_length, m_static_export_hidden};
        std::vector<float> embeds(m_static_sequence_length * m_static_export_hidden);
        std::copy(context.embeds.begin(), context.embeds.end(), embeds.begin());
        const auto embed_type = m_static_lm.get_tensor("inputs_embeds").get_element_type();
        m_static_lm.set_tensor("inputs_embeds", tensor_from_f32(embed_type, sequence_shape, embeds));

        std::vector<int64_t> positions(m_static_sequence_length);
        std::copy(context.position_ids.begin(), context.position_ids.end(), positions.begin());
        m_static_lm.set_tensor("position_ids", i64_batch_tensor(positions));

        ov::Tensor generation(ov::element::boolean, {1, m_static_sequence_length});
        std::fill_n(generation.data<bool>(), generation.get_size(), false);
        std::copy(context.generation_mask.begin(),
                  context.generation_mask.end(),
                  generation.data<bool>());
        m_static_lm.set_tensor("generation_mask", generation);

        const auto mask_type = m_static_lm.get_tensor("attention_mask").get_element_type();
        std::vector<float> mask(m_static_sequence_length * m_static_sequence_length,
                                -std::numeric_limits<float>::infinity());
        for (size_t row = 0; row < context.len; ++row) {
            std::copy(context.attention_mask.begin() + row * context.len,
                      context.attention_mask.begin() + (row + 1) * context.len,
                      mask.begin() + row * m_static_sequence_length);
        }
        m_static_lm.set_tensor("attention_mask",
                               tensor_from_f32(mask_type,
                                               {1, 1, m_static_sequence_length, m_static_sequence_length},
                                               mask));

        const auto t0 = Clock::now();
        m_static_lm.infer();
        StaticOutputs outputs;
        outputs.logits = tensor_to_f32(m_static_lm.get_output_tensor(0));
        outputs.velocity = tensor_to_f32(m_static_lm.get_output_tensor(1));
        record("lm.static", t0);
        return outputs;
    }

    std::vector<float> static_vision(const ov::Tensor& image) {
        const auto img = ImageTransform{980, 224, static_cast<int>(m_vit_patch)}(image);
        const size_t h = img.get_shape()[1], w = img.get_shape()[2];
        const size_t grid_h = h / m_vit_patch, grid_w = w / m_vit_patch;
        const size_t tokens = grid_h * grid_w;
        OPENVINO_ASSERT(tokens <= m_static_vision_tokens,
                        "BAGEL image has too many vision tokens: ",
                        tokens,
                        " > ",
                        m_static_vision_tokens);
        const auto patches = patchify(img.data<const float>(), 3, h, w, m_vit_patch, grid_h, grid_w);
        const auto patch_shape = m_static_vit.get_tensor("packed_pixel_values").get_shape();
        std::vector<float> padded(patch_shape[1] * patch_shape[2]);
        std::copy(patches.data<const float>(), patches.data<const float>() + patches.get_size(), padded.begin());
        const auto pixel_type = m_static_vit.get_tensor("packed_pixel_values").get_element_type();
        m_static_vit.set_tensor("packed_pixel_values",
                                tensor_from_f32(pixel_type, patch_shape, padded));

        std::vector<int64_t> positions = flat_position_ids(grid_h, grid_w, m_vit_max_side);
        positions.resize(patch_shape[1]);
        m_static_vit.set_tensor("position_ids", i64_batch_tensor(positions));
        const auto t0 = Clock::now();
        m_static_vit.infer();
        const auto output = tensor_to_f32(m_static_vit.get_output_tensor());
        record("vit", t0);
        return {output.begin(), output.begin() + tokens * m_static_export_hidden};
    }

    std::vector<float> static_latent_embeddings(const ov::Tensor& latent,
                                                 const std::vector<float>& timesteps,
                                                 const std::vector<int64_t>& positions) {
        const size_t tokens = latent.get_shape()[0];
        OPENVINO_ASSERT(tokens <= m_static_max_latent_tokens,
                        "BAGEL latent sequence exceeds exported limit: ",
                        tokens,
                        " > ",
                        m_static_max_latent_tokens);
        const auto latent_shape = m_static_latent_emb.get_tensor("latent").get_shape();
        const size_t dim = latent_shape[2];
        std::vector<float> padded(latent_shape[1] * dim);
        const auto latent_values = tensor_to_f32(latent);
        std::copy(latent_values.begin(), latent_values.end(), padded.begin());
        const auto latent_type = m_static_latent_emb.get_tensor("latent").get_element_type();
        m_static_latent_emb.set_tensor("latent", tensor_from_f32(latent_type, latent_shape, padded));

        std::vector<float> padded_timesteps(latent_shape[1]);
        std::copy(timesteps.begin(), timesteps.end(), padded_timesteps.begin());
        m_static_latent_emb.set_tensor("timesteps",
                                       tensor_from_f32(m_static_latent_emb.get_tensor("timesteps").get_element_type(),
                                                       {1, latent_shape[1]},
                                                       padded_timesteps));
        std::vector<int64_t> padded_positions(latent_shape[1]);
        std::copy(positions.begin(), positions.end(), padded_positions.begin());
        m_static_latent_emb.set_tensor("position_ids", i64_batch_tensor(padded_positions));
        const auto t0 = Clock::now();
        m_static_latent_emb.infer();
        const auto output = tensor_to_f32(m_static_latent_emb.get_output_tensor());
        record("latent_emb", t0);
        return {output.begin(), output.begin() + tokens * m_static_export_hidden};
    }

    std::vector<float> static_velocity(const ov::Tensor& latent,
                                       float timestep,
                                       const StaticContext& context,
                                       size_t h,
                                       size_t w) {
        const size_t tokens = latent.get_shape()[0];
        const auto latent_positions = vae_position_ids(h, w);
        const auto latent_embeds =
            static_latent_embeddings(latent, std::vector<float>(tokens, timestep), latent_positions);
        StaticContext generation = context;
        static_append_generation(generation, latent_embeds);
        const auto outputs = static_forward(generation);
        const size_t start = (context.len + 1) * m_static_patch_latent_dim;
        return {outputs.velocity.begin() + start, outputs.velocity.begin() + start + tokens * m_static_patch_latent_dim};
    }

    void static_prefill_image(StaticContext& context, const ov::Tensor& image) {
        static_append_understanding(context, static_vision(image));
    }

    StaticContext static_prefill_vae(const ov::Tensor& image, size_t& h, size_t& w) {
        const auto img = ImageTransform{static_cast<int>(m_static_image_width),
                                        static_cast<int>(m_static_image_width),
                                        16}(image);
        OPENVINO_ASSERT(img.get_shape()[1] == m_static_image_height && img.get_shape()[2] == m_static_image_width,
                        "BAGEL static VAE expects a ",
                        m_static_image_height,
                        "x",
                        m_static_image_width,
                        " image");
        ov::Tensor batched(ov::element::f32, {1, 3, m_static_image_height, m_static_image_width}, img.data());
        const auto t0 = Clock::now();
        m_static_vae_enc.set_tensor("pixel_values", batched);
        m_static_vae_enc.infer();
        const auto latent = m_static_vae_enc.get_output_tensor();
        record("vae_enc", t0);
        const auto shape = latent.get_shape();
        h = shape[2] / m_latent_patch;
        w = shape[3] / m_latent_patch;
        const auto packed = patchify(latent.data<const float>(), m_latent_channel, shape[2], shape[3], m_latent_patch, h, w);
        StaticContext context;
        static_append_generation(context,
                                 static_latent_embeddings(packed,
                                                           std::vector<float>(h * w, 0.0f),
                                                           vae_position_ids(h, w)));
        return context;
    }

    ov::Tensor static_decode(const ov::Tensor& latent, size_t h, size_t w) {
        const size_t expected_h = h * m_latent_patch;
        const size_t expected_w = w * m_latent_patch;
        OPENVINO_ASSERT(expected_h == m_static_latent_height && expected_w == m_static_latent_width,
                        "BAGEL static VAE supports only its exported latent resolution");
        ov::Tensor image_latent(ov::element::f32,
                                {1, m_latent_channel, expected_h, expected_w});
        float* destination = image_latent.data<float>();
        const float* source = latent.data<const float>();
        for (size_t gh = 0; gh < h; ++gh) {
            for (size_t gw = 0; gw < w; ++gw) {
                for (size_t py = 0; py < m_latent_patch; ++py) {
                    for (size_t px = 0; px < m_latent_patch; ++px) {
                        for (size_t channel = 0; channel < m_latent_channel; ++channel) {
                            destination[(channel * expected_h + gh * m_latent_patch + py) * expected_w +
                                        gw * m_latent_patch + px] = *source++;
                        }
                    }
                }
            }
        }
        const auto t0 = Clock::now();
        m_static_vae_dec.set_tensor("latents", image_latent);
        m_static_vae_dec.infer();
        const auto output = tensor_to_f32(m_static_vae_dec.get_output_tensor());
        record("vae_dec", t0);
        ov::Tensor result(ov::element::u8, {1, m_static_image_height, m_static_image_width, 3});
        uint8_t* result_data = result.data<uint8_t>();
        const size_t plane = m_static_image_height * m_static_image_width;
        for (size_t i = 0; i < plane; ++i) {
            for (size_t channel = 0; channel < 3; ++channel) {
                result_data[i * 3 + channel] =
                    static_cast<uint8_t>(std::clamp((output[channel * plane + i] + 1.0f) / 2.0f, 0.0f, 1.0f) *
                                         255.0f);
            }
        }
        return result;
    }

    ov::Tensor static_diffuse(size_t h,
                              size_t w,
                              const StaticContext& main,
                              const std::optional<StaticContext>& cfg_text,
                              const std::optional<StaticContext>& cfg_img,
                              const BagelImageGenerationConfig& config,
                              std::pair<float, float> default_interval) {
        const size_t tokens = h * w;
        const size_t dim = m_static_patch_latent_dim;
        ov::Tensor x(ov::element::f32, {tokens, dim});
        NumpyRandomState rng(config.seed);
        for (size_t i = 0; i < x.get_size(); ++i) {
            x.data<float>()[i] = static_cast<float>(rng.gauss());
        }

        const auto interval = config.cfg_interval.value_or(default_interval);
        const size_t steps = config.num_inference_steps;
        OPENVINO_ASSERT(steps >= 2, "num_inference_steps must be at least 2");
        std::vector<float> timesteps(steps);
        for (size_t i = 0; i < steps; ++i) {
            timesteps[i] = i + 1 == steps ? 0.0f : static_cast<float>(i * (-1.0 / (steps - 1)) + 1.0);
            timesteps[i] = config.timestep_shift * timesteps[i] /
                           (1.0f + (config.timestep_shift - 1.0f) * timesteps[i]);
        }

        for (size_t i = 0; i + 1 < steps; ++i) {
            const auto t0 = Clock::now();
            const float t = timesteps[i];
            const bool use_cfg = interval.first < t && t <= interval.second;
            auto velocity = static_velocity(x, t, main, h, w);
            if (use_cfg) {
                auto guided = velocity;
                if (cfg_text && config.cfg_text_scale > 1.0f) {
                    const auto unconditional = static_velocity(x, t, *cfg_text, h, w);
                    for (size_t k = 0; k < guided.size(); ++k) {
                        guided[k] = unconditional[k] + config.cfg_text_scale * (velocity[k] - unconditional[k]);
                    }
                }
                if (cfg_img && config.cfg_img_scale > 1.0f) {
                    const auto unconditional = static_velocity(x, t, *cfg_img, h, w);
                    for (size_t k = 0; k < guided.size(); ++k) {
                        guided[k] = unconditional[k] + config.cfg_img_scale * (guided[k] - unconditional[k]);
                    }
                }
                const float scale = std::max(std::min(l2_norm(velocity) / (l2_norm(guided) + 1e-8f), 1.0f),
                                             config.cfg_renorm_min);
                for (size_t k = 0; k < velocity.size(); ++k) {
                    velocity[k] = guided[k] * scale;
                }
            }
            const float dt = timesteps[i] - timesteps[i + 1];
            for (size_t k = 0; k < velocity.size(); ++k) {
                x.data<float>()[k] -= velocity[k] * dt;
            }
            record(use_cfg ? "step.cfg" : "step.nocfg", t0);
        }
        return static_decode(x, h, w);
    }

    std::string generate_text_static(const std::string& prompt,
                                     const std::optional<ov::Tensor>& image,
                                     size_t max_new_tokens,
                                     const std::string& system_prompt) {
        StaticContext context;
        if (!system_prompt.empty()) {
            static_add_text(context, system_prompt);
        }
        if (image) {
            static_prefill_image(context, *image);
        }
        static_add_text(context, prompt);

        int64_t current = m_bos;
        std::vector<int64_t> output;
        for (size_t i = 0; i < max_new_tokens; ++i) {
            static_append_text(context, {current});
            const auto result = static_forward(context);
            const size_t vocab = result.logits.size() / m_static_sequence_length;
            const float* last = result.logits.data() + (context.len - 1) * vocab;
            current = static_cast<int64_t>(std::max_element(last, last + vocab) - last);
            if (current == m_eos) {
                break;
            }
            output.push_back(current);
        }
        return m_tokenizer.decode(output, {{"skip_special_tokens", false}});
    }

    ov::Tensor generate_image_static(const std::string& prompt, const BagelImageGenerationConfig& config) {
        OPENVINO_ASSERT(config.size == m_static_image_width && config.size == m_static_image_height,
                        "BAGEL static export supports only ",
                        m_static_image_width,
                        "x",
                        m_static_image_height,
                        " text-to-image output");
        StaticContext context;
        static_add_text(context, prompt);
        std::optional<StaticContext> cfg_text;
        if (config.cfg_text_scale > 1.0f) {
            cfg_text = StaticContext{};
        }
        return static_diffuse(m_static_latent_height / m_latent_patch,
                              m_static_latent_width / m_latent_patch,
                              context,
                              cfg_text,
                              std::nullopt,
                              config,
                              {0.4f, 1.0f});
    }

    ov::Tensor edit_image_static(const ov::Tensor& image,
                                 const std::string& prompt,
                                 const BagelImageGenerationConfig& config) {
        size_t h = 0, w = 0;
        const StaticContext vae_context = static_prefill_vae(image, h, w);
        StaticContext context = vae_context;
        static_prefill_image(context, image);
        const StaticContext image_context = context;
        static_add_text(context, prompt);

        std::optional<StaticContext> cfg_text, cfg_img;
        if (config.cfg_text_scale > 1.0f) {
            cfg_text = image_context;
        }
        if (config.cfg_img_scale > 1.0f) {
            StaticContext prompt_context;
            static_add_text(prompt_context, prompt);
            cfg_img = prompt_context;
        }
        return static_diffuse(h, w, context, cfg_text, cfg_img, config, {0.0f, 1.0f});
    }

    // [layer * 2 + {0: key, 1: value}], each [1, kv_heads, len, head_dim]; tensors are never written after creation
    using KVCache = std::vector<ov::Tensor>;

    struct Context {
        KVCache kv;
        int64_t rope = 0;
        size_t len = 0;
    };

    struct LmOutputs {
        ov::Tensor logits;
        ov::Tensor v_t;
        KVCache present;
    };

    KVCache empty_kv() const {
        KVCache kv;
        for (size_t i = 0; i < m_layers * 2; ++i) {
            kv.emplace_back(ov::element::f32, ov::Shape{1, m_kv_heads, 0, m_head_dim});
        }
        return kv;
    }

    void reset() {
        m_ctx = Context{empty_kv(), 0, 0};
    }

    void record(const std::string& stage, Clock::time_point t0) {
        m_perf.infer_ms.emplace_back(stage, ms_since(t0));
    }

    // Generation-free calls get one padded dummy generation token in NPUW_BAGEL
    std::string lm_variant(size_t seq, size_t gen, size_t kv) const {
        if (gen == 0) {
            seq = pick_bucket(m_seq_buckets, seq + 1);
            gen = 1;
        }
        return "language S=" + std::to_string(seq) + " gen=" + std::to_string(gen) +
               " kv=" + std::to_string(pick_bucket(m_kv_buckets, std::max<size_t>(kv, 1)));
    }

    // A not-yet-compiled NPU variant runs twice so compile and inference times split
    template <typename Fn>
    void timed(const std::string& stage, const std::string& variant, Fn&& fn) {
        if (m_npu && m_compiled.insert(variant).second) {
            auto t0 = Clock::now();
            fn();
            const float first = ms_since(t0);
            m_perf.first_call_ms += first;
            t0 = Clock::now();
            fn();
            const float infer = ms_since(t0);
            m_perf.compile_ms.emplace_back(variant, first - infer);
            m_perf.infer_ms.emplace_back(stage, infer);
            return;
        }
        const auto t0 = Clock::now();
        fn();
        record(stage, t0);
    }

    // [n, hidden] f32
    ov::Tensor embed(const std::vector<int64_t>& ids) {
        const auto t0 = Clock::now();
        m_text_emb.set_input_tensor(i64_tensor(ids));
        m_text_emb.infer();
        auto out = copy_tensor(m_text_emb.get_output_tensor());
        record("text_emb", t0);
        return out;
    }

    LmOutputs forward(const ov::Tensor& embeds,
                      const std::vector<int64_t>& und_idx,
                      const std::vector<int64_t>& gen_idx,
                      const ov::Tensor& x_t,
                      float timestep,
                      const std::vector<int64_t>& vae_pos,
                      const ov::Tensor& mask,
                      const std::vector<int64_t>& pos,
                      const KVCache& past,
                      bool need_logits,
                      bool need_kv,
                      const std::string& stage) {
        ov::Tensor ts(ov::element::f32, {1});
        ts.data<float>()[0] = timestep;
        m_lm.set_tensor("inputs_embeds", embeds);
        m_lm.set_tensor("und_index", i64_tensor(und_idx));
        m_lm.set_tensor("gen_index", i64_tensor(gen_idx));
        m_lm.set_tensor("x_t", x_t);
        m_lm.set_tensor("timestep", ts);
        m_lm.set_tensor("vae_position_ids", i64_tensor(vae_pos));
        m_lm.set_tensor("attention_mask", mask);
        m_lm.set_tensor("position_ids", i64_tensor(pos));
        for (size_t i = 0; i < m_layers; ++i) {
            m_lm.set_tensor("past_key_values." + std::to_string(i) + ".key", past[i * 2]);
            m_lm.set_tensor("past_key_values." + std::to_string(i) + ".value", past[i * 2 + 1]);
        }

        // Only the needed outputs are read: NPUW_BAGEL crops an output on first access
        timed("lm." + stage + ".infer", lm_variant(embeds.get_shape()[0], gen_idx.size(), past[0].get_shape()[2]), [&] {
            m_lm.start_async();
            m_lm.wait();
        });

        const auto t0 = Clock::now();
        LmOutputs out;
        if (need_kv) {
            for (size_t i = 0; i < m_layers; ++i) {
                out.present.push_back(copy_tensor(m_lm.get_tensor("present." + std::to_string(i) + ".key")));
                out.present.push_back(copy_tensor(m_lm.get_tensor("present." + std::to_string(i) + ".value")));
            }
        }
        if (need_logits) {
            out.logits = copy_tensor(m_lm.get_tensor("logits"));
        }
        out.v_t = copy_tensor(m_lm.get_tensor("v_t"));
        record("lm." + stage + ".outputs", t0);
        return out;
    }

    // Understanding-only call on the current context; updates its KV cache
    ov::Tensor lm(const ov::Tensor& embeds,
                  const std::vector<int64_t>& pos,
                  const ov::Tensor& mask,
                  bool need_logits,
                  const std::string& stage) {
        const auto s = static_cast<int64_t>(embeds.get_shape()[0]);
        auto out = forward(embeds,
                           iota_i64(0, s),
                           {},
                           ov::Tensor(ov::element::f32, {0, latent_dim()}),
                           0.0f,
                           {},
                           mask,
                           pos,
                           m_ctx.kv,
                           need_logits,
                           true,
                           stage);
        m_ctx.kv = std::move(out.present);
        return out.logits;
    }

    // Generation call: [soi, N latent tokens, eoi] on top of the given context
    LmOutputs gen(const ov::Tensor& x_t,
                  float t,
                  const Context& ctx,
                  const std::vector<int64_t>& vae_pos,
                  bool need_kv,
                  const std::string& stage) {
        const size_t n = x_t.get_shape()[0];
        const size_t s = n + 2;
        ov::Tensor embeds(ov::element::f32, {s, m_hidden});
        float* e = embeds.data<float>();
        std::fill_n(e, embeds.get_size(), 0.0f);
        std::copy_n(m_soi_emb.data<const float>(), m_hidden, e);
        std::copy_n(m_eoi_emb.data<const float>(), m_hidden, e + (s - 1) * m_hidden);
        return forward(embeds,
                       {0, static_cast<int64_t>(s - 1)},
                       iota_i64(1, static_cast<int64_t>(s - 1)),
                       x_t,
                       t,
                       vae_pos,
                       full_mask(s, ctx.len),
                       std::vector<int64_t>(s, ctx.rope),
                       ctx.kv,
                       false,
                       need_kv,
                       stage);
    }

    void load_soi_eoi() {
        m_soi_emb = embed({m_soi});
        m_eoi_emb = embed({m_eoi});
    }

    void prefill_text(const std::string& text) {
        const auto tokenized = m_tokenizer.encode(text).input_ids;
        const int64_t* tok = tokenized.data<const int64_t>();
        std::vector<int64_t> ids{m_bos};
        ids.insert(ids.end(), tok, tok + tokenized.get_size());
        ids.push_back(m_eos);
        const size_t t = ids.size();
        lm(embed(ids),
           iota_i64(m_ctx.rope, m_ctx.rope + static_cast<int64_t>(t)),
           causal_mask(t, m_ctx.len),
           false,
           "text_prefill");
        m_ctx.len += t;
        m_ctx.rope += static_cast<int64_t>(t);
    }

    void prefill_image(const ov::Tensor& image) {
        const auto img = ImageTransform{980, 224, static_cast<int>(m_vit_patch)}(image);
        const size_t h = img.get_shape()[1], w = img.get_shape()[2];
        const size_t gh = h / m_vit_patch, gw = w / m_vit_patch;
        const auto patches = patchify(img.data<const float>(), 3, h, w, m_vit_patch, gh, gw);
        const auto pos = i64_tensor(flat_position_ids(gh, gw, m_vit_max_side));

        timed("vit", "vision " + std::to_string(gh * gw) + " patches", [&] {
            m_vit.set_tensor("pixel_values", patches);
            m_vit.set_tensor("position_ids", pos);
            m_vit.infer();
        });
        const auto img_emb = m_vit.get_output_tensor();

        load_soi_eoi();
        const size_t n = img_emb.get_shape()[0];
        const size_t s = n + 2;
        ov::Tensor seq(ov::element::f32, {s, m_hidden});
        float* d = seq.data<float>();
        std::copy_n(m_soi_emb.data<const float>(), m_hidden, d);
        std::copy_n(img_emb.data<const float>(), n * m_hidden, d + m_hidden);
        std::copy_n(m_eoi_emb.data<const float>(), m_hidden, d + (s - 1) * m_hidden);
        lm(seq, std::vector<int64_t>(s, m_ctx.rope), full_mask(s, m_ctx.len), false, "image_prefill");
        m_ctx.len += s;
        m_ctx.rope += 1;
    }

    size_t latent_dim() const {
        return m_latent_patch * m_latent_patch * m_latent_channel;
    }

    std::vector<int64_t> vae_position_ids(size_t h, size_t w) const {
        return flat_position_ids(h, w, m_max_latent);
    }

    // Encodes the image into a [soi, latent, eoi] context; returns it with the latent grid
    std::pair<Context, std::pair<size_t, size_t>> vae_prefill(const ov::Tensor& image, const Context& ctx) {
        const auto img = ImageTransform{1024, 512, 16}(image);
        const size_t img_h = img.get_shape()[1], img_w = img.get_shape()[2];
        ov::Tensor batched(ov::element::f32, {1, 3, img_h, img_w}, img.data());
        timed("vae_enc", "vae_encoder " + std::to_string(img_h) + "x" + std::to_string(img_w), [&] {
            m_vae_enc.set_input_tensor(batched);
            m_vae_enc.infer();
        });
        const auto latent = m_vae_enc.get_output_tensor();
        const auto& ls = latent.get_shape();
        const size_t h = img_h / m_latent_downsample, w = img_w / m_latent_downsample;
        const auto x_t = patchify(latent.data<const float>(), m_latent_channel, ls[2], ls[3], m_latent_patch, h, w);

        load_soi_eoi();
        auto out = gen(x_t, 0.0f, ctx, vae_position_ids(h, w), true, "vae_prefill");
        return {Context{std::move(out.present), ctx.rope + 1, ctx.len + h * w + 2}, {h, w}};
    }

    std::vector<float> velocity(const ov::Tensor& x_t,
                                float t,
                                const Context& cond,
                                const std::vector<int64_t>& vae_pos) {
        const auto v = gen(x_t, t, cond, vae_pos, false, "denoise").v_t;
        return {v.data<const float>(), v.data<const float>() + v.get_size()};
    }

    ov::Tensor diffuse(size_t h,
                       size_t w,
                       const Context& main,
                       const std::optional<Context>& cfg_text,
                       const std::optional<Context>& cfg_img,
                       const BagelImageGenerationConfig& config,
                       std::pair<float, float> default_interval) {
        const size_t n = h * w;
        const size_t dim = latent_dim();
        const auto vae_pos = vae_position_ids(h, w);
        const auto interval = config.cfg_interval.value_or(default_interval);
        load_soi_eoi();

        ov::Tensor x(ov::element::f32, {n, dim});
        float* xd = x.data<float>();
        NumpyRandomState rng(config.seed);
        for (size_t i = 0; i < n * dim; ++i) {
            xd[i] = static_cast<float>(rng.gauss());
        }

        // np.linspace(1, 0, steps, dtype=float32) with the timestep shift applied in f32
        const size_t steps = config.num_inference_steps;
        OPENVINO_ASSERT(steps >= 2, "num_inference_steps must be at least 2");
        std::vector<float> ts(steps);
        for (size_t i = 0; i < steps; ++i) {
            ts[i] = i + 1 == steps ? 0.0f : static_cast<float>(i * (-1.0 / (steps - 1)) + 1.0);
            const float shift = config.timestep_shift;
            ts[i] = shift * ts[i] / (1.0f + (shift - 1.0f) * ts[i]);
        }

        for (size_t i = 0; i + 1 < steps; ++i) {
            const auto t0 = Clock::now();
            const float first0 = m_perf.first_call_ms;
            const float t = ts[i];
            const bool use_cfg = interval.first < t && t <= interval.second;
            auto v = velocity(x, t, main, vae_pos);
            if (use_cfg) {
                auto v_ = v;
                if (cfg_text && config.cfg_text_scale > 1.0f) {
                    const auto vt = velocity(x, t, *cfg_text, vae_pos);
                    for (size_t k = 0; k < v.size(); ++k) {
                        v_[k] = vt[k] + config.cfg_text_scale * (v[k] - vt[k]);
                    }
                }
                if (cfg_img && config.cfg_img_scale > 1.0f) {
                    const auto vi = velocity(x, t, *cfg_img, vae_pos);
                    for (size_t k = 0; k < v.size(); ++k) {
                        v_[k] = vi[k] + config.cfg_img_scale * (v_[k] - vi[k]);
                    }
                }
                const float scale = std::max(std::min(l2_norm(v) / (l2_norm(v_) + 1e-8f), 1.0f), config.cfg_renorm_min);
                for (size_t k = 0; k < v.size(); ++k) {
                    v[k] = v_[k] * scale;
                }
            }
            const float dt = ts[i] - ts[i + 1];
            for (size_t k = 0; k < v.size(); ++k) {
                xd[k] = xd[k] - v[k] * dt;
            }
            m_perf.infer_ms.emplace_back(use_cfg ? "step.cfg" : "step.nocfg",
                                         ms_since(t0) - (m_perf.first_call_ms - first0));
        }
        return decode(x, h, w);
    }

    // [n, p * p * c] latent tokens -> u8 [1, H, W, 3] image
    ov::Tensor decode(const ov::Tensor& x, size_t h, size_t w) {
        const size_t p = m_latent_patch, c = m_latent_channel;
        const size_t lh = h * p, lw = w * p;
        ov::Tensor lat(ov::element::f32, {1, c, lh, lw});
        const float* src = x.data<const float>();
        float* dst = lat.data<float>();
        for (size_t gh = 0; gh < h; ++gh) {
            for (size_t gw = 0; gw < w; ++gw) {
                for (size_t py = 0; py < p; ++py) {
                    for (size_t px = 0; px < p; ++px) {
                        for (size_t ch = 0; ch < c; ++ch) {
                            dst[(ch * lh + gh * p + py) * lw + gw * p + px] = *src++;
                        }
                    }
                }
            }
        }
        timed("vae_dec", "vae_decoder " + std::to_string(lh) + "x" + std::to_string(lw), [&] {
            m_vae_dec.set_input_tensor(lat);
            m_vae_dec.infer();
        });
        const auto img = m_vae_dec.get_output_tensor();
        const size_t oh = img.get_shape()[2], ow = img.get_shape()[3];
        ov::Tensor out(ov::element::u8, {1, oh, ow, 3});
        const float* in = img.data<const float>();
        uint8_t* o = out.data<uint8_t>();
        for (size_t i = 0; i < oh * ow; ++i) {
            for (size_t ch = 0; ch < 3; ++ch) {
                const float v = std::clamp((in[ch * oh * ow + i] + 1.0f) / 2.0f, 0.0f, 1.0f);
                o[i * 3 + ch] = static_cast<uint8_t>(v * 255.0f);
            }
        }
        return out;
    }

    bool m_npu;
    bool m_static_export = false;
    Tokenizer m_tokenizer;
    ov::InferRequest m_text_emb, m_vit, m_lm, m_vae_enc, m_vae_dec;
    ov::InferRequest m_static_text_emb,
        m_static_vit,
        m_static_latent_emb,
        m_static_lm,
        m_static_vae_enc,
        m_static_vae_dec;
    std::set<std::string> m_compiled;
    std::vector<size_t> m_seq_buckets, m_kv_buckets;
    Context m_ctx;
    ov::Tensor m_soi_emb, m_eoi_emb;

    size_t m_hidden = 0, m_layers = 0, m_kv_heads = 0, m_head_dim = 0;
    size_t m_latent_channel = 0, m_latent_patch = 0, m_latent_downsample = 0;
    size_t m_vit_patch = 0, m_vit_max_side = 0, m_max_latent = 0;
    size_t m_static_sequence_length = 0, m_static_vision_tokens = 0, m_static_max_latent_tokens = 0;
    size_t m_static_export_hidden = 0, m_static_patch_latent_dim = 0;
    size_t m_static_latent_height = 0, m_static_latent_width = 0;
    size_t m_static_image_height = 0, m_static_image_width = 0;
    int64_t m_bos = 0, m_eos = 0, m_soi = 0, m_eoi = 0;
};

BagelPipeline::BagelPipeline(const std::filesystem::path& models_dir,
                             const std::string& device,
                             const std::filesystem::path& tokenizer_dir,
                             const ov::AnyMap& properties)
    : m_impl(std::make_unique<Impl>(models_dir, device, tokenizer_dir, properties)) {}

BagelPipeline::~BagelPipeline() = default;

std::string BagelPipeline::generate_text(const std::string& prompt,
                                         const std::optional<ov::Tensor>& image,
                                         size_t max_new_tokens,
                                         const std::string& system_prompt) {
    return m_impl->generate_text(prompt, image, max_new_tokens, system_prompt);
}

ov::Tensor BagelPipeline::generate_image(const std::string& prompt, const BagelImageGenerationConfig& config) {
    return m_impl->generate_image(prompt, config);
}

ov::Tensor BagelPipeline::edit_image(const ov::Tensor& image,
                                     const std::string& prompt,
                                     const BagelImageGenerationConfig& config) {
    return m_impl->edit_image(image, prompt, config);
}

const BagelPerfMetrics& BagelPipeline::get_perf_metrics() const {
    return m_impl->m_perf;
}

void BagelPipeline::reset_perf_metrics() {
    m_impl->m_perf = {};
}

}  // namespace ov::genai
