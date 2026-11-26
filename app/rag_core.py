"""Núcleo do RAG com HyDE (Parte 1 do seminário).

Contrato usado pelo app e pelos outros membros do grupo:

    import rag_core
    rag_core.iniciar()                                  # carrega corpus, encoder e índice
    r = rag_core.buscar("sua consulta", hyde=True, k=5)
    r["docs"]       -> lista de {"rank", "id", "titulo", "texto", "score"}
    r["passagens"]  -> passagens hipotéticas geradas (vazio se hyde=False)
    texto = rag_core.responder("sua consulta", r["docs"])

Para trocar de base (por exemplo, o Quati em português), mude `corpus_dir` e `idioma`
em Config. O corpus precisa estar no formato BEIR (corpus.jsonl com _id, title e text).
"""
import hashlib
import json
import os
import time
import urllib.request
import zipfile
from dataclasses import dataclass

import numpy as np

# Os prompts ficam aqui para serem trocados sem mexer no resto do código.
# O prompt "en" é o do artigo HyDE para a SciFact. O prompt "pt" é nossa adaptação, validada no
# subconjunto do Quati (HyDE-PT ficou acima do prompt com instrução em inglês).
PERFIS = {
    "en": {
        "hyde": "Please write a scientific paper passage to support/refute the claim.\nClaim: {}\nPassage:",
        "rag": ("Using only the numbered passages below, say whether the evidence supports or refutes "
                "the claim. Answer in 2 or 3 sentences and cite the passages like [1].\n\n"
                "Passages:\n{ctx}\n\nClaim: {q}\nAnswer:"),
    },
    "pt": {
        "hyde": "Escreva um trecho de texto em português que responda à pergunta em detalhes.\nPergunta: {}\nTrecho:",
        "rag": ("Responda à pergunta usando apenas os trechos numerados abaixo. Responda em 2 ou 3 frases "
                "e cite os trechos assim: [1]. Se os trechos não bastarem, diga que não encontrou a resposta.\n\n"
                "Trechos:\n{ctx}\n\nPergunta: {q}\nResposta:"),
    },
}

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"


@dataclass
class Config:
    corpus_dir: str = "beir_data/scifact"
    encoder: str = "intfloat/multilingual-e5-small"
    q_prefix: str = "query: "        # o mE5 exige prefixos; para o Contriever use "" nos dois
    d_prefix: str = "passage: "
    normalize: bool = True           # True = cosseno; False = produto escalar (Contriever)
    llm: str = "Qwen/Qwen2.5-1.5B-Instruct"
    idioma: str = "en"               # "en" ou "pt" (chave de PERFIS)
    n_hipoteses: int = 4
    max_tokens_hipotese: int = 200
    max_tokens_resposta: int = 200
    temperatura: float = 0.7
    seed: int = 42                   # mesma consulta gera as mesmas passagens (demo reproduzível)
    max_chars_contexto: int = 700
    cache_dir: str = "rag_cache"


def baixar_beir(nome, destino="beir_data"):
    """Baixa e extrai uma base BEIR (ex.: 'scifact'). Retorna a pasta da base."""
    caminho = os.path.join(destino, nome)
    if os.path.exists(os.path.join(caminho, "corpus.jsonl")):
        return caminho
    os.makedirs(destino, exist_ok=True)
    zip_path = os.path.join(destino, f"{nome}.zip")
    if not os.path.exists(zip_path):
        try:
            urllib.request.urlretrieve(BEIR_URL.format(nome), zip_path)
        except Exception as e:
            raise RuntimeError(
                f"Falha ao baixar {BEIR_URL.format(nome)} ({e}). Baixe o zip pelo navegador, "
                f"envie para a pasta '{destino}' e rode de novo.")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(destino)
    return caminho


def ler_corpus(caminho):
    corpus = {}
    with open(os.path.join(caminho, "corpus.jsonl"), encoding="utf-8") as f:
        for linha in f:
            d = json.loads(linha)
            corpus[d["_id"]] = {"title": d.get("title", ""), "text": d["text"]}
    return corpus


class RAG:
    def __init__(self, cfg=None, encoder=None):
        self.cfg = cfg or Config()
        self._encoder = encoder      # permite injetar um encoder (testes)
        self._llm = None
        self.doc_ids, self.docs, self.embs = [], [], None

    # ---------- carregamento ----------
    @property
    def encoder(self):
        if self._encoder is None:
            import torch
            from sentence_transformers import SentenceTransformer
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._encoder = SentenceTransformer(self.cfg.encoder, device=device)
        return self._encoder

    def _codificar(self, textos, prefixo):
        return self.encoder.encode([prefixo + t for t in textos], batch_size=64, convert_to_numpy=True,
                                   normalize_embeddings=self.cfg.normalize, show_progress_bar=False)

    def carregar(self):
        corpus = ler_corpus(self.cfg.corpus_dir)
        self.doc_ids = list(corpus)
        self.docs = [corpus[d] for d in self.doc_ids]
        textos = [(d["title"] + " " + d["text"]).strip() for d in self.docs]
        chave = hashlib.sha1("|".join([self.cfg.encoder, self.cfg.d_prefix, str(self.cfg.normalize),
                                       *self.doc_ids]).encode()).hexdigest()[:16]
        os.makedirs(self.cfg.cache_dir, exist_ok=True)
        arq = os.path.join(self.cfg.cache_dir, f"embs_{chave}.npy")
        if os.path.exists(arq):
            self.embs = np.load(arq)
        else:
            self.embs = self._codificar(textos, self.cfg.d_prefix)
            np.save(arq, self.embs)
        return self

    # ---------- LLM ----------
    def _carregar_llm(self):
        if self._llm is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tok = AutoTokenizer.from_pretrained(self.cfg.llm)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32
            modelo = AutoModelForCausalLM.from_pretrained(self.cfg.llm, torch_dtype=dtype).to(device)
            modelo.eval()
            self._llm = (modelo, tok)
        return self._llm

    def _chat(self, prompt, n, max_new_tokens, amostrar):
        """Gera n textos para um prompt. É o único ponto que usa o LLM (fácil de trocar em testes)."""
        import torch
        modelo, tok = self._carregar_llm()
        texto = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                        tokenize=False, add_generation_prompt=True)
        entrada = tok(texto, return_tensors="pt").to(modelo.device)
        torch.manual_seed(self.cfg.seed)
        if amostrar:
            kw = dict(do_sample=True, temperature=self.cfg.temperatura, top_p=1.0, top_k=0,
                      num_return_sequences=n)
        else:
            kw = dict(do_sample=False)
        with torch.no_grad():
            saida = modelo.generate(**entrada, max_new_tokens=max_new_tokens,
                                    pad_token_id=tok.pad_token_id if tok.pad_token_id is not None
                                    else tok.eos_token_id, **kw)
        novos = saida[:, entrada["input_ids"].shape[1]:]
        return [t.strip() for t in tok.batch_decode(novos, skip_special_tokens=True)]

    @property
    def dispositivo(self):
        """'cuda' (GPU) ou 'cpu'. Sem GPU a geração do LLM leva minutos por consulta."""
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "desconhecido"

    # ---------- API ----------
    def gerar_passagens(self, consulta):
        prompt = PERFIS[self.cfg.idioma]["hyde"].format(consulta)
        saidas = self._chat(prompt, self.cfg.n_hipoteses, self.cfg.max_tokens_hipotese, True)
        # se o LLM devolver texto vazio, usa a própria consulta (mesma regra da baseline)
        return [s if s else consulta for s in saidas]

    def buscar(self, consulta, hyde=True, k=5):
        if self.embs is None:
            raise RuntimeError("Chame carregar() (ou rag_core.iniciar()) antes de buscar.")
        t0 = time.time()
        q = self._codificar([consulta], self.cfg.q_prefix)
        passagens, t_ger = [], 0.0
        if hyde:
            tg = time.time()
            passagens = self.gerar_passagens(consulta)
            t_ger = time.time() - tg
            H = self._codificar(passagens, self.cfg.d_prefix)
            v = (q + H.sum(axis=0, keepdims=True)) / (len(passagens) + 1)   # mesma regra do hyde.py
        else:
            v = q
        scores = (v @ self.embs.T)[0]
        topo = np.argsort(-scores)[:k]
        docs = [{"rank": i + 1, "id": self.doc_ids[j], "titulo": self.docs[j]["title"],
                 "texto": self.docs[j]["text"], "score": float(scores[j])} for i, j in enumerate(topo)]
        return {"docs": docs, "passagens": passagens,
                "tempo_busca_s": time.time() - t0 - t_ger, "tempo_geracao_s": t_ger}

    def responder(self, consulta, docs):
        ctx = "\n\n".join(f"[{d['rank']}] {d['titulo']}. {d['texto'][:self.cfg.max_chars_contexto]}"
                          for d in docs)
        prompt = PERFIS[self.cfg.idioma]["rag"].format(ctx=ctx, q=consulta)
        return self._chat(prompt, 1, self.cfg.max_tokens_resposta, False)[0]


# ---------- interface de módulo (o contrato simples) ----------
_rag = None


def iniciar(cfg=None, baixar=None):
    """Carrega tudo. Se baixar='scifact', baixa a base BEIR antes."""
    global _rag
    cfg = cfg or Config()
    if baixar:
        baixar_beir(baixar, os.path.dirname(cfg.corpus_dir) or "beir_data")
    _rag = RAG(cfg).carregar()
    return _rag


def buscar(consulta, hyde=True, k=5):
    return _rag.buscar(consulta, hyde=hyde, k=k)


def responder(consulta, docs):
    return _rag.responder(consulta, docs)


def dispositivo():
    return _rag.dispositivo
