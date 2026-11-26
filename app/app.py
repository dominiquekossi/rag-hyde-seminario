"""App web do seminário: busca com e sem HyDE lado a lado, mais resposta gerada (RAG).

Uso (depois de rag_core.iniciar()):
    import app
    app.demo.launch(share=True)
"""
import html

import gradio as gr

import rag_core

CSS = """
<style>
.cards{display:flex;flex-direction:column;gap:10px}
.card{border:1px solid #d9d9d9;border-radius:10px;padding:10px 14px;background:#fff;color:#222}
.card .t{font-weight:600;margin:0 0 4px 0}
.card .m{font-size:12px;color:#666;margin-bottom:6px}
.card .x{font-size:14px;line-height:1.45}
.badge{display:inline-block;font-size:11px;padding:1px 8px;border-radius:999px;margin-left:6px;font-weight:600}
.novo{background:#dff5e3;color:#14602a}
.saiu{background:#fdeacc;color:#7a4a00}
.hip{border-left:4px solid #6b7fd7;padding:8px 12px;margin:0 0 8px 0;background:#f4f5fb;color:#222;font-size:14px;border-radius:0 8px 8px 0;white-space:pre-wrap}
.vazio{color:#777;font-style:italic}
</style>
"""

EXEMPLOS = [
    "Vitamin D supplementation reduces the risk of bone fractures.",
    "Statins lower the risk of cardiovascular disease.",
    "Smoking increases the risk of lung cancer.",
    "Regular exercise improves insulin sensitivity.",
]


def _cards(docs, outros_ids, selo, classe, rotulo_vazio="Nenhum resultado."):
    if not docs:
        return f"<p class='vazio'>{rotulo_vazio}</p>"
    partes = []
    for d in docs:
        badge = f"<span class='badge {classe}'>{selo}</span>" if d["id"] not in outros_ids else ""
        trecho = html.escape(d["texto"][:320]) + ("..." if len(d["texto"]) > 320 else "")
        partes.append(
            f"<div class='card'><p class='t'>{d['rank']}. {html.escape(d['titulo'] or d['id'])}{badge}</p>"
            f"<div class='m'>id {html.escape(d['id'])} · similaridade {d['score']:.3f}</div>"
            f"<div class='x'>{trecho}</div></div>")
    return "<div class='cards'>" + "".join(partes) + "</div>"


def pesquisar(consulta, k, gerar_resposta):
    consulta = (consulta or "").strip()
    if not consulta:
        vazio = "<p class='vazio'>Digite uma consulta.</p>"
        return vazio, vazio, vazio, "", ""
    k = int(k)
    denso = rag_core.buscar(consulta, hyde=False, k=k)
    hy = rag_core.buscar(consulta, hyde=True, k=k)
    ids_d = {d["id"] for d in denso["docs"]}
    ids_h = {d["id"] for d in hy["docs"]}

    html_d = _cards(denso["docs"], ids_h, "saiu com HyDE", "saiu")
    html_h = _cards(hy["docs"], ids_d, "novo com HyDE", "novo")
    html_p = "".join(f"<div class='hip'>{html.escape(p)}</div>" for p in hy["passagens"])

    resposta = ""
    t_resp = 0.0
    if gerar_resposta:
        import time
        t0 = time.time()
        resposta = rag_core.responder(consulta, hy["docs"])
        t_resp = time.time() - t0

    novos = len(ids_h - ids_d)
    disp = rag_core.dispositivo()
    aviso = ("\n\n**Atenção: sem GPU.** A geração com o LLM fica muito lenta (minutos). "
             "No Colab, ative a GPU T4 em Ambiente de execução > Alterar tipo de ambiente de execução."
             if disp == "cpu" else "")
    tempos = (f"**Dispositivo:** {disp} · **Busca densa:** {denso['tempo_busca_s']:.2f} s · "
              f"**HyDE:** geração das passagens {hy['tempo_geracao_s']:.1f} s + busca {hy['tempo_busca_s']:.2f} s"
              + (f" · **Resposta:** {t_resp:.1f} s" if gerar_resposta else "")
              + f"\n\n{novos} de {len(ids_h)} documentos do HyDE não aparecem na busca densa." + aviso)
    return html_d, html_h, html_p, resposta, tempos


def construir_demo():
    with gr.Blocks(title="RAG com HyDE") as demo:
        gr.HTML(CSS)
        gr.Markdown("# RAG com HyDE\nBusca por significado com e sem **HyDE** (documento hipotético gerado por um LLM), "
                    "lado a lado, e uma resposta gerada só com os documentos encontrados.")
        with gr.Row():
            consulta = gr.Textbox(label="Consulta", placeholder="Escreva uma afirmação ou pergunta",
                                  lines=2, scale=4)
            k = gr.Slider(1, 10, value=5, step=1, label="Documentos (k)", scale=1)
        gerar = gr.Checkbox(value=True, label="Gerar resposta (RAG)")
        botao = gr.Button("Pesquisar", variant="primary")
        gr.Examples(EXEMPLOS, inputs=consulta, label="Exemplos")
        tempos = gr.Markdown()
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Busca densa (sem HyDE)")
                saida_d = gr.HTML()
            with gr.Column():
                gr.Markdown("### Busca com HyDE")
                saida_h = gr.HTML()
        with gr.Accordion("Passagens hipotéticas geradas pelo LLM", open=False):
            saida_p = gr.HTML()
        gr.Markdown("### Resposta gerada (usa os documentos do HyDE)")
        saida_r = gr.Textbox(label="Resposta", lines=5, interactive=False)

        entradas = [consulta, k, gerar]
        saidas = [saida_d, saida_h, saida_p, saida_r, tempos]
        botao.click(pesquisar, entradas, saidas)
        consulta.submit(pesquisar, entradas, saidas)
    return demo


demo = construir_demo()
