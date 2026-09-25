"""
src/generate_linkedin_pdf.py

Gera o documento PDF profissional (carrossel de slides) para publicação no LinkedIn:
docs/ufc_fighting_styles_linkedin_carousel.pdf

Formato: Quadrado (1:1), 300 DPI, Design UFC Dark Octagon (#080D1A), 
perfeito para o leitor de documentos do LinkedIn (Mobile & Desktop).
"""
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

OUTPUT_PDF = "docs/ufc_fighting_styles_linkedin_carousel.pdf"
os.makedirs("docs", exist_ok=True)

print("Iniciando geração do PDF Carrossel para LinkedIn...")

with PdfPages(OUTPUT_PDF) as pdf:
    # -------------------------------------------------------------
    # SLIDE 1: SCORECARD GERAL & AUDITORIA TÉCNICA (GOLDEN SET)
    # -------------------------------------------------------------
    print("Processando Slide 1: Golden Set Scorecard...")
    img1 = Image.open("docs/images/golden_set_scorecard.png")
    
    fig1 = plt.figure(figsize=(14.0, 14.4), dpi=300)
    fig1.patch.set_facecolor("#080D1A")
    ax1 = fig1.add_subplot(111)
    ax1.axis("off")
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 100)

    # Imagem completa do Scorecard ocupando quase 100% da tela para máxima legibilidade
    ax1.imshow(img1, extent=[0.5, 99.5, 0.5, 99.5], aspect="auto", interpolation="lanczos", zorder=2)
    pdf.savefig(fig1, facecolor="#080D1A", dpi=300)
    plt.close(fig1)

    # -------------------------------------------------------------
    # SLIDE 2: PROJEÇÃO PCA & FRONTEIRAS DE DECISÃO
    # -------------------------------------------------------------
    print("Processando Slide 2: Projeção PCA e Espaço Latente...")
    img2 = Image.open("docs/images/cluster_pca_scatter.png")

    fig2 = plt.figure(figsize=(14.0, 14.0), dpi=300)
    fig2.patch.set_facecolor("#080D1A")
    ax2 = fig2.add_subplot(111)
    ax2.axis("off")
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 100)

    # Header de Paginação
    bar2 = FancyBboxPatch((2.0, 95.8), 96.0, 3.2, boxstyle="round,pad=0.2,rounding_size=0.5",
                          facecolor="#111A2E", edgecolor="#243350", linewidth=0.8)
    ax2.add_patch(bar2)
    ax2.text(3.5, 97.4, "SLIDE 02 / 05  •  ESPAÇO LATENTE & FRONTEIRAS DE DECISÃO",
             fontsize=8.5, fontweight="bold", color="#34D399", va="center")
    ax2.text(96.5, 97.4, "DESLIZE PARA O LADO →", fontsize=8.0, fontweight="bold", color="#94A3B8", ha="right", va="center")

    # Título do Slide
    ax2.text(3.5, 93.4, "Agrupamento dos 3.400+ Lutadores no Espaço PCA (61.9% da Variância)",
             fontsize=13.5, fontweight="bold", color="#F8FAFC", va="center")
    ax2.text(3.5, 90.8, "Como o K-Means e a fronteira euclidiana separam os polos e mapeiam lutadores híbridos (Well-rounded)",
             fontsize=9.0, color="#94A3B8", va="center")

    # Moldura e Imagem central (Aspect Ratio ~1.47: 92w x 62.5h)
    frame2 = FancyBboxPatch((3.5, 18.0), 93.0, 70.0, boxstyle="round,pad=0.3,rounding_size=0.8",
                            facecolor="#FFFFFF", edgecolor="#34D399", linewidth=1.4, zorder=1)
    ax2.add_patch(frame2)
    ax2.imshow(img2, extent=[4.5, 95.5, 19.0, 87.0], aspect="auto", interpolation="lanczos", zorder=3)

    # Callout inferior de Key Takeaway
    callout2 = FancyBboxPatch((2.0, 2.0), 96.0, 14.0, boxstyle="round,pad=0.3,rounding_size=0.8",
                              facecolor="#0F172A", edgecolor="#34D399", linewidth=1.1, zorder=4)
    ax2.add_patch(callout2)
    accent2 = Rectangle((2.5, 15.6), 95.0, 0.4, facecolor="#34D399", edgecolor="none", zorder=5)
    ax2.add_patch(accent2)
    ax2.text(4.0, 13.0, "▶ INSIGHT DE ENGENHARIA DE MACHINE LEARNING:", fontsize=9.2, fontweight="bold", color="#34D399", va="center", zorder=5)
    ax2.text(4.0, 10.4, "• O modelo identifica não apenas clusters rígidos, mas uma FRONTEIRA EUCLIDIANA (|d_strik - d_wrest| < 0.5).", fontsize=8.4, color="#E2E8F0", va="center", zorder=5)
    ax2.text(4.0, 7.8, "• É nessa zona de transição que residem os atletas modernos mais dominantes do esporte (GSP, Jon Jones, Usman e Makhachev).", fontsize=8.4, color="#94A3B8", va="center", zorder=5)
    ax2.text(4.0, 5.2, "• Atletas sem volume ofensivo e com absorção extrema (CM Punk, Moutinho) convergem de forma isolada no Cluster 0.", fontsize=8.4, color="#FCA5A5", va="center", zorder=5)

    pdf.savefig(fig2, facecolor="#080D1A", dpi=300)
    plt.close(fig2)

    # -------------------------------------------------------------
    # SLIDE 3: ASSINATURA ESTATÍSTICA POR ESTILO
    # -------------------------------------------------------------
    print("Processando Slide 3: Assinatura Estatística dos Clusters...")
    img3 = Image.open("docs/images/cluster_profiles_comparison.png")

    fig3 = plt.figure(figsize=(14.0, 14.0), dpi=300)
    fig3.patch.set_facecolor("#080D1A")
    ax3 = fig3.add_subplot(111)
    ax3.axis("off")
    ax3.set_xlim(0, 100)
    ax3.set_ylim(0, 100)

    # Header de Paginação
    bar3 = FancyBboxPatch((2.0, 95.8), 96.0, 3.2, boxstyle="round,pad=0.2,rounding_size=0.5",
                          facecolor="#111A2E", edgecolor="#243350", linewidth=0.8)
    ax3.add_patch(bar3)
    ax3.text(3.5, 97.4, "SLIDE 03 / 05  •  ASSINATURA ESTATÍSTICA DOS ESTILOS",
             fontsize=8.5, fontweight="bold", color="#A78BFA", va="center")
    ax3.text(96.5, 97.4, "DESLIZE PARA O LADO →", fontsize=8.0, fontweight="bold", color="#94A3B8", ha="right", va="center")

    # Título do Slide
    ax3.text(3.5, 93.4, "Perfil Comparativo Médio dos 4 Estilos Fundamentais",
             fontsize=13.5, fontweight="bold", color="#F8FAFC", va="center")
    ax3.text(3.5, 90.8, "Volume de golpes, absorção de dano por minuto, quedas aplicadas e taxa de finalização",
             fontsize=9.0, color="#94A3B8", va="center")

    # Moldura e Imagem central (Aspect Ratio ~1.50: 92w x 61.5h)
    frame3 = FancyBboxPatch((3.5, 18.0), 93.0, 70.0, boxstyle="round,pad=0.3,rounding_size=0.8",
                            facecolor="#FFFFFF", edgecolor="#A78BFA", linewidth=1.4, zorder=1)
    ax3.add_patch(frame3)
    ax3.imshow(img3, extent=[4.5, 95.5, 19.0, 87.0], aspect="auto", interpolation="lanczos", zorder=3)

    # Callout inferior de Key Takeaway
    callout3 = FancyBboxPatch((2.0, 2.0), 96.0, 14.0, boxstyle="round,pad=0.3,rounding_size=0.8",
                              facecolor="#0F172A", edgecolor="#A78BFA", linewidth=1.1, zorder=4)
    ax3.add_patch(callout3)
    accent3 = Rectangle((2.5, 15.6), 95.0, 0.4, facecolor="#A78BFA", edgecolor="none", zorder=5)
    ax3.add_patch(accent3)
    ax3.text(4.0, 13.0, "▶ MÉTRICAS QUE SEPARAM OS PERFIS NO OCTÓGONO:", fontsize=9.2, fontweight="bold", color="#A78BFA", va="center", zorder=5)
    ax3.text(4.0, 10.4, "• STRIKER: Lidera em golpes conectados (3.68/min) com taxa de absorção controlada (3.46/min).", fontsize=8.4, color="#E2E8F0", va="center", zorder=5)
    ax3.text(4.0, 7.8, "• WRESTLER: Aplica 3.69 quedas/15m e mantém a menor taxa de dano absorvido da organização (2.77/min).", fontsize=8.4, color="#94A3B8", va="center", zorder=5)
    ax3.text(4.0, 5.2, "• SUBMISSION SPECIALIST: Pico absoluto de tentativas de finalização (11.01/15m), mas volume de golpes reduzido.", fontsize=8.4, color="#C4B5FD", va="center", zorder=5)

    pdf.savefig(fig3, facecolor="#080D1A", dpi=300)
    plt.close(fig3)

    # -------------------------------------------------------------
    # SLIDE 4: VALIDAÇÃO MATEMÁTICA DE HIPERPARÂMETROS
    # -------------------------------------------------------------
    print("Processando Slide 4: Otimização de Hiperparâmetros (Elbow & Silhouette)...")
    img4 = Image.open("docs/images/elbow_silhouette.png")

    fig4 = plt.figure(figsize=(14.0, 14.0), dpi=300)
    fig4.patch.set_facecolor("#080D1A")
    ax4 = fig4.add_subplot(111)
    ax4.axis("off")
    ax4.set_xlim(0, 100)
    ax4.set_ylim(0, 100)

    # Header de Paginação
    bar4 = FancyBboxPatch((2.0, 95.8), 96.0, 3.2, boxstyle="round,pad=0.2,rounding_size=0.5",
                          facecolor="#111A2E", edgecolor="#243350", linewidth=0.8)
    ax4.add_patch(bar4)
    ax4.text(3.5, 97.4, "SLIDE 04 / 05  •  RIGOR ESTATÍSTICO & OTIMIZAÇÃO",
             fontsize=8.5, fontweight="bold", color="#F59E0B", va="center")
    ax4.text(96.5, 97.4, "DESLIZE PARA O LADO →", fontsize=8.0, fontweight="bold", color="#94A3B8", ha="right", va="center")

    # Título do Slide
    ax4.text(3.5, 93.4, "Seleção Ótima de k=4: Método Elbow e Coeficiente de Silhouette",
             fontsize=13.5, fontweight="bold", color="#F8FAFC", va="center")
    ax4.text(3.5, 90.8, "Validação quantitativa que impede a criação de clusters artificiais ou overfitting da partição",
             fontsize=9.0, color="#94A3B8", va="center")

    # Moldura e Imagem central (Aspect Ratio ~1.80: 92w x 60h)
    frame4 = FancyBboxPatch((3.5, 18.0), 93.0, 70.0, boxstyle="round,pad=0.3,rounding_size=0.8",
                            facecolor="#FFFFFF", edgecolor="#F59E0B", linewidth=1.4, zorder=1)
    ax4.add_patch(frame4)
    ax4.imshow(img4, extent=[4.5, 95.5, 23.0, 83.0], aspect="auto", interpolation="lanczos", zorder=3)

    # Callout inferior de Key Takeaway
    callout4 = FancyBboxPatch((2.0, 2.0), 96.0, 14.0, boxstyle="round,pad=0.3,rounding_size=0.8",
                              facecolor="#0F172A", edgecolor="#F59E0B", linewidth=1.1, zorder=4)
    ax4.add_patch(callout4)
    accent4 = Rectangle((2.5, 15.6), 95.0, 0.4, facecolor="#F59E0B", edgecolor="none", zorder=5)
    ax4.add_patch(accent4)
    ax4.text(4.0, 13.0, "▶ JUSTIFICATIVA MATEMÁTICA DA ARQUITETURA:", fontsize=9.2, fontweight="bold", color="#F59E0B", va="center", zorder=5)
    ax4.text(4.0, 10.4, "• Ponto do Cotovelo (Elbow): A maior redução relativa de inércia intra-cluster ocorre exatamente em k=4.", fontsize=8.4, color="#E2E8F0", va="center", zorder=5)
    ax4.text(4.0, 7.8, "• Silhouette Score Máximo (0.224): k=4 maximiza a coesão interna sem dispersão de amostras no espaço PCA.", fontsize=8.4, color="#94A3B8", va="center", zorder=5)
    ax4.text(4.0, 5.2, "• Testes com k=5 e k=6 degradaram o Silhouette Score para 0.189 e criaram clusters redundantes.", fontsize=8.4, color="#FDE68A", va="center", zorder=5)

    pdf.savefig(fig4, facecolor="#080D1A", dpi=300)
    plt.close(fig4)

    # -------------------------------------------------------------
    # SLIDE 5: LIÇÕES DE ML, LIMITAÇÕES & MLOPS (CONCLUSÃO)
    # -------------------------------------------------------------
    print("Processando Slide 5: Síntese de Auditoria e Lições de ML...")
    fig5 = plt.figure(figsize=(14.0, 14.0), dpi=300)
    fig5.patch.set_facecolor("#080D1A")
    ax5 = fig5.add_subplot(111)
    ax5.axis("off")
    ax5.set_xlim(0, 100)
    ax5.set_ylim(0, 100)

    # Header de Paginação
    bar5 = FancyBboxPatch((2.0, 95.8), 96.0, 3.2, boxstyle="round,pad=0.2,rounding_size=0.5",
                          facecolor="#111A2E", edgecolor="#243350", linewidth=0.8)
    ax5.add_patch(bar5)
    ax5.text(3.5, 97.4, "SLIDE 05 / 05  •  CONCLUSÕES & ENGENHARIA DE DADOS",
             fontsize=8.5, fontweight="bold", color="#38BDF8", va="center")
    ax5.text(96.5, 97.4, "REPOSITÓRIO NO GITHUB ★", fontsize=8.0, fontweight="bold", color="#38BDF8", ha="right", va="center")

    # Título do Slide
    ax5.text(3.5, 93.0, "O Que Aprendemos Auditando Modelos de Machine Learning no UFC",
             fontsize=14.0, fontweight="bold", color="#F8FAFC", va="center")
    ax5.text(3.5, 90.6, "Principais lições sobre viés de dados, auditoria supervisionada e limites de dados públicos",
             fontsize=9.2, color="#94A3B8", va="center")

    # 3 Cards executivos de lições aprendidas
    cards_lessons = [
        {
            "y": 66.0, "h": 22.0,
            "title": "1. O Algoritmo Avalia o Comportamento no Octógono, Não a Reputação",
            "tag": "VIÉS DE DOMÍNIO",
            "tag_color": "#EF4444", "tag_bg": "#7F1D1D",
            "bullets": [
                ("Yoel Romero (Medalhista Olímpico):", "Historicamente um dos maiores wrestlers do mundo, mas no UFC quase não aplicava quedas (<1.8/15min) e lutava quase exclusivamente como striker. O modelo previu Striker."),
                ("Gilbert Burns & Demian Maia (BJJ):", "Contra wrestlers de ponta (Woodley, Usman), foram obrigados a passar rounds em pé. Suas médias históricas no UFCStats foram puxadas para Striker.")
            ]
        },
        {
            "y": 42.0, "h": 22.0,
            "title": "2. O Limite do Algoritmo é a Granularidade do Dataset (Gaps do UFCStats)",
            "tag": "GAPS DE DADOS",
            "tag_color": "#F59E0B", "tag_bg": "#78350F",
            "bullets": [
                ("Falta de Tempo de Controle (Control Time):", "Duração em solo e grade resolveria de imediato as classificações de especialistas em Jiu-Jitsu que controlam o adversário sem bater muito."),
                ("Falta de Potência de Golpe (Power Index):", "Separaria brawlers resistentes que nocauteiam com um golpe (Tai Tuivasa) de atletas com pouco output e sem pegada (CM Punk, Moutinho).")
            ]
        },
        {
            "y": 18.0, "h": 22.0,
            "title": "3. Golden Sets Supervisionados Devem Ser Padrão em Modelos Não-Supervisionados",
            "tag": "BOA PRÁTICA DE ML",
            "tag_color": "#10B981", "tag_bg": "#064E3B",
            "bullets": [
                ("Nunca confiar apenas em Silhouette Score:", "Métricas de validação interna não sabem quem é Alex Poatan ou Khabib. O Golden Set supervisionado revelou a taxa real de 80.0% de precisão de domínio."),
                ("Engenharia de Fronteiras Híbridas:", "A regra euclidiana de transição suave (|d_strik - d_wrest| < 0.5) capturou perfeitamente os atletas Well-rounded sem necessidade de re-treinar o K-Means.")
            ]
        }
    ]

    for c in cards_lessons:
        cy, ch = c["y"], c["h"]
        c_bg = FancyBboxPatch((2.0, cy), 96.0, ch, boxstyle="round,pad=0.3,rounding_size=0.8",
                              facecolor="#0F172A", edgecolor="#243350", linewidth=1.1)
        ax5.add_patch(c_bg)

        # Header do card
        ax5.text(3.8, cy + ch - 2.8, c["title"], fontsize=9.4, fontweight="bold", color="#F8FAFC", va="center")
        
        # Tag
        c_tag = FancyBboxPatch((81.5, cy + ch - 3.8), 15.0, 2.2, boxstyle="round,pad=0.1,rounding_size=0.4",
                               facecolor=c["tag_bg"], edgecolor=c["tag_color"], linewidth=0.8)
        ax5.add_patch(c_tag)
        ax5.text(89.0, cy + ch - 2.7, c["tag"], fontsize=6.8, fontweight="bold", color="#FFFFFF", ha="center", va="center")

        # Linha interna
        ax5.add_patch(Rectangle((3.8, cy + ch - 4.8), 92.4, 0.08, facecolor="#1E293B", edgecolor="none"))

        # Bullets
        by = cy + ch - 7.6
        for b_title, b_desc in c["bullets"]:
            ax5.text(4.0, by, f"• {b_title}", fontsize=8.2, fontweight="bold", color=c["tag_color"], va="center")
            by -= 2.3
            words = b_desc.split(" ")
            lines = []
            cur = ""
            for w in words:
                if len(cur + " " + w) < 92:
                    cur = (cur + " " + w).strip()
                else:
                    lines.append(cur)
                    cur = w
            if cur:
                lines.append(cur)
            for line_str in lines:
                ax5.text(5.5, by, line_str, fontsize=7.8, color="#94A3B8", va="center")
                by -= 2.0
            by -= 1.0

    # Banner Final de GitHub / MLOps
    cta_bg = FancyBboxPatch((2.0, 2.0), 96.0, 13.5, boxstyle="round,pad=0.3,rounding_size=0.8",
                            facecolor="#111C33", edgecolor="#38BDF8", linewidth=1.2)
    ax5.add_patch(cta_bg)
    ax5.text(4.0, 12.0, "★ PIPELINE COMPLETO DE MLOPS DISPONÍVEL NO GITHUB", fontsize=10.0, fontweight="bold", color="#38BDF8", va="center")
    ax5.text(4.0, 9.4, "• Arquitetura com Apache Airflow (DAG), Docker, Scikit-learn, Feature Engineering e Golden Set Audit.", fontsize=8.2, color="#E2E8F0", va="center")
    ax5.text(4.0, 7.0, "• O pipeline orquestra desde o scraping e pré-processamento até o retreinamento com Grid Search de pesos e export de predições.", fontsize=8.2, color="#94A3B8", va="center")
    ax5.text(4.0, 4.4, "★ Veja o repositório completo com todos os códigos, notebooks e documentação no link do post!", fontsize=8.4, fontweight="semibold", color="#FBBF24", va="center")

    pdf.savefig(fig5, facecolor="#080D1A", dpi=300)
    plt.close(fig5)

print(f"Sucesso! PDF Carrossel para LinkedIn gerado em: {OUTPUT_PDF}")

# -------------------------------------------------------------
# EXPORTAÇÃO DOS SLIDES EM IMAGENS PNG INDIVIDUAIS (PARA CARROSSEL DE IMAGENS)
# -------------------------------------------------------------
import fitz
print("Exportando páginas do PDF como imagens PNG de alta resolução...")
doc = fitz.open(OUTPUT_PDF)
for page_num in range(len(doc)):
    page = doc.load_page(page_num)
    pix = page.get_pixmap(dpi=200)
    out_img = f"docs/images/linkedin_slide_{page_num + 1}.png"
    pix.save(out_img)
    print(f"Slide {page_num + 1} exportado para: {out_img}")
doc.close()
print("Todas as imagens individuais dos slides foram salvas em docs/images/!")
