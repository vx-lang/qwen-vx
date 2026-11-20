// Record what the model does on each prompt, for simulate_drafts.py: the greedy output, and the
// experts that the router picks at every position in every layer. It uses llama.cpp's library on
// the GGUF file of the model, so it needs no Vx.
//
//   record_experts MODEL.gguf OUT_DIR MAX_NEW_TOKENS PROMPT_FILE...
//
// For a prompt file p.txt (the full text, chat template included), it writes:
//   OUT_DIR/p.tokens  int32: the number of prompt tokens, then every token (prompt and output)
//   OUT_DIR/p.experts int32: [position][layer][8], the router's choices in order of weight, for
//                     the positions from the last prompt token to the token before the last one
// and once: OUT_DIR/vocab.bin, for each token id an int32 length and the bytes of its text.
//
// Build it with the llama.cpp of Homebrew (0.7.0 when this was written):
//
//   clang++ -std=c++17 -O2 scripts/record_experts.cpp -I/opt/homebrew/include \
//       -L/opt/homebrew/lib -lllama -lggml -lggml-base -Wl,-rpath,/opt/homebrew/lib \
//       -o record_experts
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

struct Recorder {
  int layers = 0, used = 0;
  // experts[layer] holds the choices of positions 0, 1, ... in order.
  std::vector<std::vector<int32_t>> experts;
  std::vector<uint8_t> scratch;
  bool sawNames = false;
};

static bool isTopK(const char *name, int *layer) {
  const char *p = "ffn_moe_topk-";
  if (strncmp(name, p, strlen(p)) != 0)
    return false;
  *layer = atoi(name + strlen(p));
  return true;
}

static bool onTensor(struct ggml_tensor *t, bool ask, void *data) {
  Recorder *r = (Recorder *)data;
  int layer = -1;
  if (!isTopK(t->name, &layer))
    return ask ? false : true;
  if (ask)
    return true;
  if (t->type != GGML_TYPE_I32 || layer < 0 || layer >= r->layers) {
    fprintf(stderr, "unexpected tensor %s type %d\n", t->name, (int)t->type);
    exit(1);
  }
  int used = (int)t->ne[0], n = (int)t->ne[1];
  if (r->used == 0)
    r->used = used;
  if (used != r->used) {
    fprintf(stderr, "%s has %d experts, expected %d\n", t->name, used, r->used);
    exit(1);
  }
  size_t bytes = ggml_nbytes(t);
  r->scratch.resize(bytes);
  ggml_backend_tensor_get(t, r->scratch.data(), 0, bytes);
  for (int j = 0; j < n; j++) {
    const int32_t *row = (const int32_t *)(r->scratch.data() + j * t->nb[1]);
    for (int e = 0; e < used; e++)
      r->experts[layer].push_back(row[e * (t->nb[0] / 4)]);
  }
  return true;
}

static std::string readFile(const char *path) {
  std::ifstream f(path, std::ios::binary);
  std::stringstream s;
  s << f.rdbuf();
  return s.str();
}

static std::string baseName(const std::string &path) {
  size_t slash = path.find_last_of('/');
  std::string b = slash == std::string::npos ? path : path.substr(slash + 1);
  size_t dot = b.find_last_of('.');
  return dot == std::string::npos ? b : b.substr(0, dot);
}

int main(int argc, char **argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: record_experts MODEL.gguf OUT_DIR MAX_NEW_TOKENS PROMPT_FILE...\n");
    return 1;
  }
  std::string out = argv[2];
  int maxNew = atoi(argv[3]);
  llama_backend_init();
  llama_model_params mp = llama_model_default_params();
  mp.n_gpu_layers = 999;
  llama_model *model = llama_model_load_from_file(argv[1], mp);
  if (!model)
    return 1;
  const llama_vocab *vocab = llama_model_get_vocab(model);
  int nVocab = llama_vocab_n_tokens(vocab);
  {
    FILE *f = fopen((out + "/vocab.bin").c_str(), "wb");
    std::vector<char> buf(1024);
    for (int id = 0; id < nVocab; id++) {
      int32_t len = llama_token_to_piece(vocab, id, buf.data(), (int32_t)buf.size(), 0, true);
      if (len < 0)
        len = 0;
      fwrite(&len, 4, 1, f);
      fwrite(buf.data(), 1, len, f);
    }
    fclose(f);
  }
  Recorder rec;
  rec.layers = llama_model_n_layer(model);
  llama_context_params cp = llama_context_default_params();
  cp.n_ctx = 12288;
  cp.n_batch = 512;
  cp.n_ubatch = 512;
  cp.cb_eval = onTensor;
  cp.cb_eval_user_data = &rec;
  llama_context *ctx = llama_init_from_model(model, cp);
  if (!ctx)
    return 1;
  for (int a = 4; a < argc; a++) {
    std::string text = readFile(argv[a]);
    std::vector<llama_token> tokens(text.size() + 16);
    int n = llama_tokenize(vocab, text.c_str(), (int32_t)text.size(), tokens.data(),
                           (int32_t)tokens.size(), false, true);
    if (n < 0) {
      fprintf(stderr, "tokenize failed for %s\n", argv[a]);
      return 1;
    }
    tokens.resize(n);
    int promptLen = n;
    llama_memory_clear(llama_get_memory(ctx), true);
    rec.experts.assign(rec.layers, {});
    int decoded = 0;
    for (int at = 0; at < promptLen; at += 512) {
      int len = std::min(512, promptLen - at);
      if (llama_decode(ctx, llama_batch_get_one(tokens.data() + at, len)) != 0) {
        fprintf(stderr, "decode failed\n");
        return 1;
      }
      decoded += len;
    }
    for (int g = 0; g < maxNew; g++) {
      const float *logits = llama_get_logits_ith(ctx, -1);
      llama_token best = 0;
      for (int id = 1; id < nVocab; id++)
        if (logits[id] > logits[best])
          best = id;
      tokens.push_back(best);
      if (llama_vocab_is_eog(vocab, best))
        break;
      if (llama_decode(ctx, llama_batch_get_one(&tokens.back(), 1)) != 0) {
        fprintf(stderr, "decode failed\n");
        return 1;
      }
      decoded++;
    }
    // llama.cpp runs the last layer only for the tokens whose logits it returns, so every layer
    // has the positions from the last prompt token on. The simulator needs those, up to the
    // token before the last one.
    int ran = (int)tokens.size() - 1;
    int first = promptLen - 1, count = ran - first;
    for (int l = 0; l < rec.layers; l++)
      if ((int)rec.experts[l].size() < (decoded - first) * rec.used) {
        fprintf(stderr, "layer %d has %zu choices, expected at least %d\n", l,
                rec.experts[l].size(), (decoded - first) * rec.used);
        return 1;
      }
    std::string base = out + "/" + baseName(argv[a]);
    FILE *f = fopen((base + ".tokens").c_str(), "wb");
    fwrite(&promptLen, 4, 1, f);
    fwrite(tokens.data(), 4, tokens.size(), f);
    fclose(f);
    f = fopen((base + ".experts").c_str(), "wb");
    for (int p = 0; p < count; p++)
      for (int l = 0; l < rec.layers; l++) {
        // The rows of a layer end at position decoded - 1.
        size_t row = rec.experts[l].size() / rec.used - (size_t)(decoded - first) + p;
        fwrite(rec.experts[l].data() + row * rec.used, 4, rec.used, f);
      }
    fclose(f);
    printf("%s: %d prompt tokens, %d new, %d layers x %d experts\n", baseName(argv[a]).c_str(),
           promptLen, (int)tokens.size() - promptLen, rec.layers, rec.used);
    fflush(stdout);
  }
  llama_free(ctx);
  llama_model_free(model);
  return 0;
}
