# Code pour analyser la qualite du ciel sur des images nocturnes, en se basant uniquement sur des metriques photometriques
# Code composé d'algorythmes originaux et tiré de banques de donnée Open scources tel que github.com
# Code commenté annoté par Claude AI pour une meilleur comprehension
# Interface utilisateur et style ergonomique généré par Intelligence artificielle.

"""
Analyseur de qualite du ciel — Score photometrique relatif
===========================================================
Approche purement photometrique, sans reference SQM ni constantes calibrees.
Toutes les metriques sont normalisees sur le lot charge : la meilleure image
du lot obtient toujours le score le plus bas, les autres sont positives.

Metriques calculees :
  1. bg_mean       — luminosite moyenne du fond (etoiles effacees par blur median)
  2. veil_index    — fraction spatiale du fond au-dessus d'un seuil adaptatif
  3. contrast_ratio — rapport P5(fond) / P95(image brute), mesure combien
                      le fond "monte" par rapport aux etoiles faibles.
                      Proche de 0 = bon contraste ; proche de 1 = fond noye.
  4. skew          — asymetrie de l'histogramme vers les hautes valeurs.
                      0 = distribution symetrique/sombre ; valeur elevee =
                      queue lumineuse caracteristique de pollution diffuse.

Score de pollution (0 = ciel propre, 100 = tres pollue) :
  Moyenne ponderee des metriques normalisees min-max sur le lot :
    score = 0.45 * norm(bg_mean)
          + 0.25 * norm(veil_index)
          + 0.20 * norm(contrast_ratio)
          + 0.10 * norm(skew)
  Recalcule a chaque ajout/suppression d'image.
"""

import sys
import csv
import numpy as np
import cv2
from scipy.stats import skew as scipy_skew

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QListWidget, QSplitter,
    QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QCheckBox, QGroupBox, QScrollArea, QFrame, QTextEdit
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar
)
from matplotlib.figure import Figure


# ===============================================================
#  CONSTANTES
# ===============================================================

BLUR_KERNEL    = 21    # noyau median blur pour effacer les etoiles
VEIL_THRESHOLD = 0.15  # fraction du max de bg_blur comme seuil adaptatif

# Poids du score final (somme = 1.0)
WEIGHTS = {
    'bg_mean':        0.45,
    'veil_index':     0.25,
    'contrast_ratio': 0.20,
    'skew':           0.10,
}


# ===============================================================
#  MODELE DE DONNEES
# ===============================================================

class ImageMetrics:
    __slots__ = (
        'filepath', 'filename', 'gray',
        'bg_mean', 'veil_index', 'contrast_ratio', 'skew',
        'score',
    )

    def __init__(self, filepath: str, gray: np.ndarray):
        self.filepath       = filepath
        self.filename       = filepath.replace('\\', '/').split('/')[-1]
        self.gray           = gray
        self.bg_mean        = 0.0
        self.veil_index     = 0.0
        self.contrast_ratio = 0.0
        self.skew           = 0.0
        self.score          = 0.0


# ===============================================================
#  CALCUL DES METRIQUES
# ===============================================================

def compute_raw(m: ImageMetrics) -> None:
    """
    Calcule les 4 metriques brutes. Ne calcule pas le score
    (qui depend de l'ensemble du lot).
    """
    gray_f = m.gray.astype(np.float32)
    bg_blur = cv2.medianBlur(m.gray, BLUR_KERNEL).astype(np.float32)

    # 1. bg_mean : luminosite moyenne du fond
    m.bg_mean = float(np.mean(bg_blur))

    # 2. veil_index : fraction de pixels du fond au-dessus d'un seuil adaptatif
    #    Seuil = VEIL_THRESHOLD * max(bg_blur), evite de dependre d'une echelle fixe
    threshold = VEIL_THRESHOLD * float(np.max(bg_blur))
    m.veil_index = float(np.mean(bg_blur > threshold))

    # 3. contrast_ratio : P5(fond) / P95(image brute)
    #    Mesure l'envahissement du fond vis-a-vis des sources faibles.
    #    Si le fond est noir et les etoiles brillantes -> ratio ~ 0.
    #    Si le fond est eleve et les etoiles noyees -> ratio proche de 1.
    p5_bg  = float(np.percentile(bg_blur, 5))
    p95_im = float(np.percentile(gray_f, 95))
    m.contrast_ratio = float(np.clip(p5_bg / (p95_im + 1e-6), 0.0, 1.0))

    # 4. skew : asymetrie de l'histogramme de l'image brute
    #    Queue vers les hautes valeurs = signe de dome lumineux diffus.
    #    scipy_skew peut etre negatif sur images tres saturees -> on prend max(0).
    flat = gray_f.flatten()
    m.skew = float(max(0.0, scipy_skew(flat)))


def recompute_scores(metrics_list: list) -> None:
    """
    Normalise min-max chaque metrique sur le lot, puis calcule
    le score pondere pour chaque image.
    """
    if not metrics_list:
        return

    keys = list(WEIGHTS.keys())
    raw = {k: np.array([getattr(m, k) for m in metrics_list]) for k in keys}

    # Normalisation min-max par metrique
    norm = {}
    for k in keys:
        mn, mx = raw[k].min(), raw[k].max()
        if mx - mn < 1e-9:
            norm[k] = np.zeros(len(metrics_list))
        else:
            norm[k] = (raw[k] - mn) / (mx - mn)

    for i, m in enumerate(metrics_list):
        m.score = float(sum(WEIGHTS[k] * norm[k][i] for k in keys) * 100.0)


# ===============================================================
#  UTILITAIRES COULEUR
# ===============================================================

def score_to_color(score: float) -> str:
    if score <= 25:  return "#a6e3a1"
    if score <= 50:  return "#f9e2af"
    if score <= 75:  return "#fab387"
    return "#f38ba8"

METRIC_COLORS = {
    'bg_mean':        "#fab387",
    'veil_index':     "#89dceb",
    'contrast_ratio': "#cba6f7",
    'skew':           "#f9e2af",
}


# ===============================================================
#  WIDGET FIGURE + TOOLBAR
# ===============================================================

class FigureWithToolbar(QWidget):
    def __init__(self, figsize=(9, 5), dpi=100, parent=None):
        super().__init__(parent)
        self.figure  = Figure(figsize=figsize, dpi=dpi)
        self.canvas  = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

    def draw(self):  self.canvas.draw()
    def clear(self): self.figure.clear()


# ===============================================================
#  FENETRE PRINCIPALE
# ===============================================================

class SkyQualityAnalyzer(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Analyseur de Pollution Lumineuse — Score photometrique")
        self.setGeometry(60, 60, 1450, 880)
        self.metrics_list: list = []
        self._init_ui()
        self._apply_stylesheet()

    # ----------------------------------------------------------

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # ── Panel gauche ──────────────────────────────────────
        left = QWidget()
        left.setFixedWidth(280)
        ll = QVBoxLayout(left)
        ll.setSpacing(6)

        for text, slot in [
            ("+ Ajouter des images",     self.add_images),
            ("x Supprimer la selection", self.remove_selected),
            ("  Tout supprimer",         self.clear_all),
        ]:
            btn = QPushButton(text)
            btn.clicked.connect(slot)
            ll.addWidget(btn)

        sep = QFrame(); sep.setFrameShape(QFrame.HLine); sep.setFrameShadow(QFrame.Sunken)
        ll.addWidget(sep)
        ll.addWidget(QLabel("<b>Images chargees :</b>"))

        self.image_list = QListWidget()
        self.image_list.setSelectionMode(QListWidget.ExtendedSelection)
        ll.addWidget(self.image_list)

        filter_box = QGroupBox("Metriques affichees (barres)")
        fl = QVBoxLayout(filter_box)
        self._metric_cbs = []
        cb_defs = [
            ("Luminosite fond (bg_mean)",   "bg_mean"),
            ("Veil index",                  "veil_index"),
            ("Contrast ratio",              "contrast_ratio"),
            ("Asymetrie histogramme (skew)","skew"),
        ]
        for label, key in cb_defs:
            cb = QCheckBox(label)
            cb.setChecked(True)
            cb.stateChanged.connect(self._refresh_barchart)
            fl.addWidget(cb)
            self._metric_cbs.append((cb, key))
        ll.addWidget(filter_box)

        self.ref_label = QLabel("")
        self.ref_label.setWordWrap(True)
        self.ref_label.setStyleSheet("color: #89b4fa; font-size: 11px; padding: 4px;")
        ll.addWidget(self.ref_label)

        btn_export = QPushButton("Exporter CSV")
        btn_export.clicked.connect(self.export_csv)
        ll.addWidget(btn_export)
        ll.addStretch()

        # ── Panel droit ───────────────────────────────────────
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        self.tab_widget = QTabWidget()

        self.hist_fig = FigureWithToolbar(figsize=(10, 5))
        self.tab_widget.addTab(self.hist_fig, "Histogrammes")

        self.bar_fig = FigureWithToolbar(figsize=(10, 5))
        self.tab_widget.addTab(self.bar_fig, "Metriques comparees")

        self.score_fig = FigureWithToolbar(figsize=(10, 5))
        self.tab_widget.addTab(self.score_fig, "Scores")

        self.rank_table = self._make_rank_table()
        self.tab_widget.addTab(self.rank_table, "Classement")

        self.raw_table = self._make_raw_table()
        self.tab_widget.addTab(self.raw_table, "Valeurs brutes")

        self.tab_widget.addTab(self._make_doc_tab(), "Documentation")

        rl.addWidget(self.tab_widget)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([280, 1170])
        splitter.setCollapsible(0, False)
        root.addWidget(splitter)

    # ----------------------------------------------------------

    def _make_rank_table(self) -> QTableWidget:
        t = QTableWidget()
        t.setColumnCount(7)
        t.setHorizontalHeaderLabels([
            "Rang", "Image", "Score\n(0=propre)",
            "bg_mean", "veil_index", "contrast_ratio", "skew"
        ])
        t.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        t.setEditTriggers(QTableWidget.NoEditTriggers)
        t.setSortingEnabled(True)
        return t

    def _make_raw_table(self) -> QTableWidget:
        t = QTableWidget()
        t.setColumnCount(6)
        t.setHorizontalHeaderLabels([
            "Image", "bg_mean", "veil_index", "contrast_ratio", "skew", "score"
        ])
        t.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        t.setSortingEnabled(True)
        return t

    def _make_doc_tab(self) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(24, 20, 24, 20)

        doc_html = """
<h2 style="color:#89dceb; font-family:monospace;">Documentation — Score photometrique relatif</h2>
<p style="color:#cba6f7;">Approche purement photometrique, sans reference externe (SQM, Bortle, constantes physiques).
Toutes les valeurs sont relatives au lot d'images charge. La meilleure image = score minimal,
les autres obtiennent un score positif proportionnel a leur pollution relative.</p>
<hr style="border-color:#45475a;"/>

<h3 style="color:#fab387;">1. bg_mean — Luminosite du fond</h3>
<p><b>Calcul :</b> Filtre median 21x21 px → supprime les etoiles et sources ponctuelles.
<code>bg_mean = mean(bg_blur)</code></p>
<p><b>Interpretation :</b> Mesure directe de la brillance de surface du fond du ciel.
Valeur basse = fond noir = peu de pollution. C'est la metrique la plus determinante (poids 45%).</p>
<p><b>Poids dans le score :</b> 45%</p>

<hr style="border-color:#45475a;"/>

<h3 style="color:#89dceb;">2. veil_index — Etendue spatiale du voile</h3>
<p><b>Calcul :</b> Seuil adaptatif = 15% du maximum de bg_blur.
<code>veil_index = fraction(bg_blur > seuil)</code></p>
<p><b>Pourquoi adaptatif ?</b> Un seuil fixe (ex: 40/255) echoue sur les images sous-exposees
(tout sous le seuil → veil = 0) ou surexposees (tout dessus → veil = 1). En prenant 15% du max
local, le seuil s'adapte a la plage dynamique reelle de chaque image.</p>
<p><b>Interpretation :</b> 0.0 = dome tres localise ou inexistant. 1.0 = ciel uniformement voile.
Un dome de pollution localise donne bg_mean eleve mais veil_index faible ;
un brouillard lumineux generalise donne les deux eleves.</p>
<p><b>Poids dans le score :</b> 25%</p>

<hr style="border-color:#45475a;"/>

<h3 style="color:#cba6f7;">3. contrast_ratio — Rapport fond/etoiles faibles</h3>
<p><b>Calcul :</b> <code>contrast_ratio = P5(bg_blur) / P95(image brute)</code></p>
<p><b>Interpretation :</b> Mesure dans quelle mesure le fond du ciel "envahit" les sources faibles.
<ul>
  <li>Ratio ≈ 0 : fond tres sombre (P5 ≈ 0), etoiles faibles bien visibles (P95 eleve)</li>
  <li>Ratio ≈ 1 : fond autant ou plus lumineux que les etoiles faibles → tout est noye</li>
</ul>
C'est la metrique la plus sensible aux halos de villes eloignees qui relevent le fond
sans necessairement saturer le centre de l'image.</p>
<p><b>Poids dans le score :</b> 20%</p>

<hr style="border-color:#45475a;"/>

<h3 style="color:#f9e2af;">4. skew — Asymetrie de l'histogramme</h3>
<p><b>Calcul :</b> Coefficient d'asymetrie de Fisher (scipy.stats.skew) applique a l'histogramme
de l'image brute. Valeur = max(0, skew) pour ignorer les asymetries vers le noir.</p>
<p><b>Interpretation :</b> Un ciel propre a un histogramme pique vers les basses valeurs (fond noir)
avec une queue etroite vers les hautes valeurs (etoiles brillantes) → skew moderement positif.
Un ciel pollue a un histogramme decale et aplati vers le haut → skew plus fort et/ou repartition
plus etale, signature de la remontee diffuse du fond.</p>
<p><b>Note :</b> Cette metrique est complementaire de bg_mean : deux images peuvent avoir le meme
fond moyen mais des distributions tres differentes.</p>
<p><b>Poids dans le score :</b> 10%</p>

<hr style="border-color:#45475a;"/>

<h3 style="color:#89b4fa; font-size:16px;">Score final (0-100, relatif au lot)</h3>
<pre style="background:#181825; padding:12px; border-radius:4px; color:#a6e3a1; font-family:monospace;">
Etape 1 — Normalisation min-max par metrique sur le lot :
  norm(x) = (x - min_lot) / (max_lot - min_lot)

Etape 2 — Combinaison ponderee :
  score = 100 x (0.45 x norm(bg_mean)
               + 0.25 x norm(veil_index)
               + 0.20 x norm(contrast_ratio)
               + 0.10 x norm(skew))

Resultat :
  - Image la moins polluee du lot → score ≈ 0
  - Image la plus polluee du lot  → score = 100
  - Les autres → valeurs intermediaires proportionnelles
</pre>
<p style="color:#f38ba8;"><b>Limites :</b> le score est relatif au lot. Ajouter une image tres polluee
ecrasera les differences entre les bonnes images. Pour un tri efficace, charger des images
du meme setup et de conditions variees. Le score ne dit pas si le ciel est objectivement bon
ou mauvais — seulement qui est meilleur que qui dans votre lot.</p>
<br/>
<p style="color:#6c7086; font-size:11px;">
Methode photometrique — aucune reference externe (Unihedron, Bortle, etc.).
Normalisations internes uniquement.
</p>
"""
        text_widget = QTextEdit()
        text_widget.setReadOnly(True)
        text_widget.setHtml(doc_html)
        text_widget.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e2e; color: #cdd6f4;
                font-family: 'Segoe UI', sans-serif; font-size: 13px;
                border: none; padding: 8px;
            }
        """)
        layout.addWidget(text_widget)
        scroll.setWidget(inner)
        return scroll

    def _apply_stylesheet(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color:#1e1e2e; color:#cdd6f4;
                font-family:'Consolas','Courier New',monospace; font-size:12px; }
            QGroupBox { font-weight:bold; border:1px solid #45475a; border-radius:6px;
                margin-top:8px; padding-top:4px; }
            QGroupBox::title { subcontrol-origin:margin; left:8px; color:#89b4fa; }
            QPushButton { background-color:#313244; color:#cdd6f4; border:1px solid #45475a;
                padding:6px 10px; border-radius:4px; }
            QPushButton:hover   { background-color:#45475a; }
            QPushButton:pressed { background-color:#585b70; }
            QListWidget { background-color:#181825; border:1px solid #45475a; border-radius:4px; }
            QListWidget::item:selected { background-color:#313244; }
            QTableWidget { background-color:#181825; gridline-color:#45475a; border:1px solid #45475a; }
            QHeaderView::section { background-color:#313244; color:#89b4fa; padding:4px; border:none; font-weight:bold; }
            QTabWidget::pane { border:1px solid #45475a; }
            QTabBar::tab { background:#313244; color:#cdd6f4; padding:6px 14px;
                border-top-left-radius:4px; border-top-right-radius:4px; }
            QTabBar::tab:selected { background:#45475a; color:#89dceb; }
            QCheckBox { spacing:6px; }
            QCheckBox::indicator { width:14px; height:14px; border:1px solid #45475a;
                border-radius:3px; background:#181825; }
            QCheckBox::indicator:checked { background:#89b4fa; }
            QSplitter::handle { background:#45475a; width:2px; }
            QScrollArea { border:none; }
        """)

    # ── Gestion des images ────────────────────────────────────

    def add_images(self):
        filepaths, _ = QFileDialog.getOpenFileNames(
            self, "Selectionner des images du ciel nocturne",
            "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        for fp in filepaths:
            img_bgr = cv2.imread(fp)
            if img_bgr is None:
                self.statusBar().showMessage(f"Impossible de lire : {fp}", 4000)
                continue
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            m = ImageMetrics(fp, gray)
            compute_raw(m)
            self.metrics_list.append(m)
            self.image_list.addItem(m.filename)
        recompute_scores(self.metrics_list)
        self._update_all()

    def remove_selected(self):
        rows = sorted(
            {self.image_list.row(it) for it in self.image_list.selectedItems()},
            reverse=True
        )
        for row in rows:
            del self.metrics_list[row]
            self.image_list.takeItem(row)
        recompute_scores(self.metrics_list)
        self._update_all()

    def clear_all(self):
        self.metrics_list.clear()
        self.image_list.clear()
        self._update_all()

    def _update_all(self):
        if self.metrics_list:
            best = min(self.metrics_list, key=lambda m: m.score)
            self.ref_label.setText(
                f"Meilleure image du lot : {best.filename}\n"
                f"bg={best.bg_mean:.1f}  vi={best.veil_index:.3f}  "
                f"cr={best.contrast_ratio:.3f}  sk={best.skew:.2f}"
            )
        else:
            self.ref_label.setText("")
        self._refresh_histograms()
        self._refresh_barchart()
        self._refresh_score_chart()
        self._refresh_rank_table()
        self._refresh_raw_table()

    # ── Onglet 1 : Histogrammes ───────────────────────────────

    def _refresh_histograms(self):
        self.hist_fig.clear()
        fig = self.hist_fig.figure
        fig.patch.set_facecolor("#1e1e2e")
        if not self.metrics_list:
            self.hist_fig.draw(); return

        ax = fig.add_subplot(111)
        ax.set_facecolor("#181825")
        palette = plt.cm.tab20(np.linspace(0, 1, len(self.metrics_list)))

        for i, m in enumerate(self.metrics_list):
            hist = cv2.calcHist([m.gray], [0], None, [256], [0, 256]).flatten()
            hist_smooth = np.convolve(hist, np.ones(5)/5, mode='same')
            label = (f"{m.filename}  |  bg={m.bg_mean:.1f}  "
                     f"vi={m.veil_index:.3f}  cr={m.contrast_ratio:.3f}  "
                     f"sk={m.skew:.2f}  score={m.score:.1f}")
            ax.plot(hist_smooth, color=palette[i], label=label, alpha=0.85, linewidth=1.5)
            ax.axvline(m.bg_mean, color=palette[i], linestyle='--', alpha=0.35, linewidth=0.8)

        ax.set_title("Histogrammes (lissage 5 pts | pointille = bg_mean)",
                     color="#89dceb", fontsize=11, pad=10)
        ax.set_xlabel("Intensite pixel (0=noir  255=blanc)", color="#a6adc8")
        ax.set_ylabel("Nombre de pixels", color="#a6adc8")
        ax.tick_params(colors="#a6adc8")
        for sp in ax.spines.values(): sp.set_edgecolor("#45475a")
        ax.grid(True, alpha=0.15, color="#585b70")
        ax.legend(loc='upper right', fontsize=7.5,
                  facecolor="#313244", edgecolor="#45475a", labelcolor="#cdd6f4")
        fig.tight_layout()
        self.hist_fig.draw()

    # ── Onglet 2 : Barres comparatives ───────────────────────

    def _refresh_barchart(self):
        self.bar_fig.clear()
        fig = self.bar_fig.figure
        fig.patch.set_facecolor("#1e1e2e")
        if not self.metrics_list:
            self.bar_fig.draw(); return

        active = []
        for cb, key in self._metric_cbs:
            if cb.isChecked():
                values = [getattr(m, key) for m in self.metrics_list]
                active.append((key, values, METRIC_COLORS[key]))

        if not active:
            self.bar_fig.draw(); return

        ax = fig.add_subplot(111)
        ax.set_facecolor("#181825")
        n, k = len(self.metrics_list), len(active)
        width   = min(0.8 / k, 0.22)
        indices = np.arange(n)

        def normalize(arr):
            a = np.array(arr, dtype=float)
            mn, mx = a.min(), a.max()
            return np.full_like(a, 0.5) if mx - mn < 1e-9 else (a - mn) / (mx - mn)

        for j, (key, values, color) in enumerate(active):
            offset    = (j - (k-1)/2) * width
            norm_vals = normalize(values)
            bars = ax.bar(indices + offset, norm_vals, width,
                          label=key, color=color, alpha=0.85, zorder=2)
            for bar, v in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.015,
                        f"{v:.3f}", ha='center', va='bottom',
                        fontsize=6.5, color="#cdd6f4", zorder=3, rotation=40)

        ax.set_xticks(indices)
        xlabels = [f"{m.filename}\nscore={m.score:.1f}" for m in self.metrics_list]
        ax.set_xticklabels(xlabels, rotation=30, ha='right', fontsize=8, color="#a6adc8")
        ax.set_ylabel("Valeur normalisee [0=min du lot, 1=max] — brutes annotees", color="#a6adc8")
        ax.set_title("Comparaison des metriques (valeurs brutes annotees, normalisees pour l'affichage)",
                     color="#89dceb", fontsize=11, pad=10)
        ax.set_ylim(0, 1.40)
        ax.tick_params(colors="#a6adc8")
        ax.grid(True, axis='y', alpha=0.15, color="#585b70", zorder=0)
        for sp in ax.spines.values(): sp.set_edgecolor("#45475a")
        ax.legend(facecolor="#313244", edgecolor="#45475a",
                  labelcolor="#cdd6f4", fontsize=8, loc='upper right')
        fig.tight_layout()
        self.bar_fig.draw()

    # ── Onglet 3 : Scores ─────────────────────────────────────

    def _refresh_score_chart(self):
        self.score_fig.clear()
        fig = self.score_fig.figure
        fig.patch.set_facecolor("#1e1e2e")
        if not self.metrics_list:
            self.score_fig.draw(); return

        # Subplot 1 : barres de score
        ax1 = fig.add_subplot(121)
        ax1.set_facecolor("#181825")
        ranked = sorted(self.metrics_list, key=lambda m: m.score)
        colors = [score_to_color(m.score) for m in ranked]
        scores = [m.score for m in ranked]
        indices = np.arange(len(ranked))
        bars = ax1.barh(indices, scores, color=colors, alpha=0.85, zorder=2)
        for bar, m in zip(bars, ranked):
            ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                     f"{m.score:.1f}", va='center', fontsize=8, color="#cdd6f4")
        ax1.set_yticks(indices)
        ax1.set_yticklabels([m.filename for m in ranked], fontsize=8, color="#a6adc8")
        ax1.set_xlabel("Score de pollution (0=propre, relatif au lot)", color="#a6adc8")
        ax1.set_title("Classement des images", color="#89dceb", fontsize=10, pad=8)
        ax1.set_xlim(0, 115)
        ax1.tick_params(colors="#a6adc8")
        ax1.grid(True, axis='x', alpha=0.15, color="#585b70", zorder=0)
        for sp in ax1.spines.values(): sp.set_edgecolor("#45475a")

        # Subplot 2 : decomposition du score par metrique (barres empilees normalisees)
        ax2 = fig.add_subplot(122)
        ax2.set_facecolor("#181825")
        keys   = list(WEIGHTS.keys())
        colors2 = [METRIC_COLORS[k] for k in keys]

        # Recalcul des contributions individuelles normalisees
        raw_arr = {k: np.array([getattr(m, k) for m in ranked], dtype=float) for k in keys}
        norm_arr = {}
        for k in keys:
            mn, mx = raw_arr[k].min(), raw_arr[k].max()
            norm_arr[k] = np.zeros(len(ranked)) if mx - mn < 1e-9 else (raw_arr[k] - mn) / (mx - mn)

        bottoms = np.zeros(len(ranked))
        for k, c in zip(keys, colors2):
            contrib = norm_arr[k] * WEIGHTS[k] * 100
            ax2.barh(indices, contrib, left=bottoms, color=c, alpha=0.85, label=k, zorder=2)
            bottoms += contrib

        ax2.set_yticks(indices)
        ax2.set_yticklabels([m.filename for m in ranked], fontsize=8, color="#a6adc8")
        ax2.set_xlabel("Contribution au score (empilee)", color="#a6adc8")
        ax2.set_title("Decomposition par metrique", color="#89dceb", fontsize=10, pad=8)
        ax2.tick_params(colors="#a6adc8")
        ax2.grid(True, axis='x', alpha=0.15, color="#585b70", zorder=0)
        for sp in ax2.spines.values(): sp.set_edgecolor("#45475a")
        ax2.legend(facecolor="#313244", edgecolor="#45475a",
                   labelcolor="#cdd6f4", fontsize=8, loc='lower right')

        fig.tight_layout()
        self.score_fig.draw()

    # ── Onglet 4 : Classement ─────────────────────────────────

    def _refresh_rank_table(self):
        t = self.rank_table
        t.setSortingEnabled(False)
        ranked = sorted(self.metrics_list, key=lambda m: m.score)
        t.setRowCount(len(ranked))

        for row, m in enumerate(ranked):
            sc = score_to_color(m.score)

            def cell(text, fg=None, align=Qt.AlignCenter):
                item = QTableWidgetItem(text)
                item.setTextAlignment(align)
                if fg:
                    q = QColor(fg); item.setForeground(q)
                    b = QColor(fg); b.setAlpha(40); item.setBackground(b)
                return item

            t.setItem(row, 0, cell(f"#{row+1}", fg=sc))
            t.setItem(row, 1, QTableWidgetItem(m.filename))
            t.setItem(row, 2, cell(f"{m.score:.1f}", fg=sc))
            t.setItem(row, 3, cell(f"{m.bg_mean:.2f}",
                                   fg=METRIC_COLORS['bg_mean']))
            t.setItem(row, 4, cell(f"{m.veil_index:.4f}",
                                   fg=METRIC_COLORS['veil_index']))
            t.setItem(row, 5, cell(f"{m.contrast_ratio:.4f}",
                                   fg=METRIC_COLORS['contrast_ratio']))
            t.setItem(row, 6, cell(f"{m.skew:.3f}",
                                   fg=METRIC_COLORS['skew']))

        t.setSortingEnabled(True)
        t.resizeRowsToContents()

    # ── Onglet 5 : Valeurs brutes ─────────────────────────────

    def _refresh_raw_table(self):
        t = self.raw_table
        t.setSortingEnabled(False)
        t.setRowCount(len(self.metrics_list))
        for i, m in enumerate(self.metrics_list):
            vals = [m.filename,
                    f"{m.bg_mean:.2f}",
                    f"{m.veil_index:.4f}",
                    f"{m.contrast_ratio:.4f}",
                    f"{m.skew:.4f}",
                    f"{m.score:.1f}"]
            for col, v in enumerate(vals):
                item = QTableWidgetItem(v)
                if col == 5: item.setForeground(QColor(score_to_color(m.score)))
                if col == 1: item.setForeground(QColor(METRIC_COLORS['bg_mean']))
                if col == 2: item.setForeground(QColor(METRIC_COLORS['veil_index']))
                if col == 3: item.setForeground(QColor(METRIC_COLORS['contrast_ratio']))
                if col == 4: item.setForeground(QColor(METRIC_COLORS['skew']))
                t.setItem(i, col, item)
        t.setSortingEnabled(True)

    # ── Export CSV ────────────────────────────────────────────

    def export_csv(self):
        if not self.metrics_list:
            self.statusBar().showMessage("Aucune image a exporter.", 3000); return
        fp, _ = QFileDialog.getSaveFileName(
            self, "Exporter CSV", "metriques_ciel.csv", "CSV (*.csv)")
        if not fp: return
        ranked = sorted(self.metrics_list, key=lambda m: m.score)
        with open(fp, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(["rang", "fichier", "bg_mean", "veil_index",
                        "contrast_ratio", "skew", "score"])
            for rank, m in enumerate(ranked, 1):
                w.writerow([rank, m.filename,
                            f"{m.bg_mean:.4f}", f"{m.veil_index:.4f}",
                            f"{m.contrast_ratio:.4f}", f"{m.skew:.4f}",
                            f"{m.score:.2f}"])
        self.statusBar().showMessage(f"Exporte : {fp}", 4000)


# ===============================================================
#  POINT D'ENTREE
# ===============================================================

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = SkyQualityAnalyzer()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()