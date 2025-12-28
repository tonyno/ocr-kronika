# Technický Popis Pipeline pro Digitalizaci a Optimalizaci Rukopisů

## Přehled Architektury

Tento systém představuje sofistikovanou třístupňovou pipeline pro transformaci fyzických dokumentů do profesionálně vypadajících digitálních PDF dokumentů. Každý krok využívá pokročilé algoritmy počítačového vidění a matematické transformace, které společně vytvářejí výsledek na úrovni komerčních OCR systémů.

---

## Fáze 1: Inteligentní Organizace a Normalizace Dat (`01_sort.py`)

### Koncepce
První fáze implementuje **deterministický algoritmus pro normalizaci názvů souborů** s využitím lexikografického řazení. Systém transformuje chaotickou kolekci skenovaných obrázků do strukturované posloupnosti s konzistentními identifikátory.

### Technické Detaily
- **Lexikografické řazení**: Využívá Pythonův nativní `sorted()` algoritmus, který garantuje stabilní pořadí podle Unicode hodnot znaků
- **Deterministické přejmenování**: Generuje sekvenční identifikátory ve formátu `page{index:03d}a.jpg`, kde `03d` zajišťuje správné lexikografické řazení i při velkém počtu souborů
- **Metadata preservation**: `shutil.copy2()` zachovává všechny metadatové informace včetně časových značek, což umožňuje pozdější analýzu nebo audit trail

**Výsledek**: Deterministic mapping mezi originálními soubory a normalizovanými identifikátory, který eliminuje nejednoznačnosti v dalších fázích zpracování.

---

## Fáze 2: Pokročilé Počítačové Vidění pro Čištění Pozadí (`02_clean_bg.py`)

### Koncepce
Druhá fáze představuje **multi-stage image processing pipeline** založenou na pokročilých algoritmech počítačového vidění. Na rozdíl od jednoduchých thresholding metod, tento systém využívá **adaptivní lokální thresholding** kombinovaný s **morfologickými operacemi** a **komponentovou analýzou** pro dosažení profesionálních výsledků.

### Technické Detaily

#### 2.1 Normalizace Kontrastu
- **Min-Max normalizace**: `cv2.normalize()` s `NORM_MINMAX` transformuje pixelové hodnoty do plného rozsahu [0, 255]
- **Adaptace na různé světelné podmínky**: Kompenzuje variabilitu v expozici mezi jednotlivými skeny

#### 2.2 Adaptivní Lokální Thresholding (Kritická Komponenta)
**Toto je srdce celého systému** - na rozdíl od globálního thresholdingu, který selhává při nerovnoměrném osvětlení, používáme **Gaussian-based adaptive thresholding**:

```python
cv2.adaptiveThreshold(
    gray,
    maxValue=255,
    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    thresholdType=cv2.THRESH_BINARY,
    blockSize=83,
    C=9
)
```

**Jak to funguje:**
- **Lokální analýza**: Algoritmus rozdělí obraz na překrývající se bloky o velikosti 83×83 pixelů
- **Gaussovská vážená průměrování**: Pro každý pixel se vypočítá vážený průměr sousedních pixelů pomocí Gaussovského jádra (blíže pixely mají větší váhu)
- **Dynamický threshold**: Každý pixel je klasifikován jako text/pozadí na základě lokálního průměru minus konstanta C (9)
- **Výhoda**: Automaticky se přizpůsobí lokálním změnám jasu, stínům a nerovnoměrnému osvětlení

**Matematicky**: Pro pixel `(x,y)`:
```
threshold(x,y) = weighted_mean(neighborhood(x,y)) - C
pixel(x,y) = 255 if gray(x,y) > threshold(x,y), else 0
```

#### 2.3 Morfologické Operace - Dilation
- **Strukturní element**: 2×2 kernel pro expanzi textových oblastí
- **Účel**: Zesílení tenkých tahů pera pro lepší tiskovou kvalitu
- **Matematický základ**: Erozní-dilatační operace založené na teorii množin

#### 2.4 Komponentová Analýza (Connected Components Analysis)
**Toto je pokročilá technika pro eliminaci šumu:**

```python
nb_components, output, stats, _ = cv2.connectedComponentsWithStats(
    255 - bw, connectivity=8
)
```

**Algoritmus:**
- **8-connectivity**: Analyzuje propojení pixelů v 8 směrech (včetně diagonál)
- **Statistická analýza**: Pro každou komponentu vypočítá velikost, bounding box, centroid
- **Filtrace podle velikosti**: Komponenty menší než 90 pixelů jsou klasifikovány jako šum a odstraněny
- **Zachování struktury**: Větší komponenty (skutečný text) jsou zachovány

**Výsledek**: Eliminace artefaktů, prachu a skenovacích chyb při zachování všech relevantních textových prvků.

#### 2.5 Anti-aliasing pomocí Gaussovského Rozmazání
- **Gaussovské jádro 3×3**: Vytváří plynulé přechody na hranách textu
- **Účel**: Eliminace pixelizace a vytvoření profesionálního vzhledu
- **Matematický základ**: Konvoluce s 2D Gaussovskou funkcí pro sub-pixelovou přesnost

### Proč NENÍ toto Neuronová Síť?

**Důležitá poznámka**: OpenCV (`cv2`) v tomto kódu **NEPOUŽÍVÁ neuronové sítě**. Místo toho využívá **tradiční algoritmy počítačového vidění** založené na:

1. **Matematické morfologii** (morfologické operace)
2. **Statistické analýze** (lokální průměry, variance)
3. **Grafové teorii** (connected components)
4. **Konvolučních operacích** (Gaussovské filtry)

Tyto metody jsou **deterministické**, **vysvětlitelné** a **výpočetně efektivní**. Na rozdíl od neuronových sítí, které jsou "černé skříňky", každý krok tohoto algoritmu má jasný matematický základ a lze ho přesně analyzovat.

**Výhody tohoto přístupu:**
- ✅ **Determinismus**: Stejný vstup vždy produkuje stejný výstup
- ✅ **Rychlost**: Žádné GPU, žádné trénování, okamžité výsledky
- ✅ **Transparentnost**: Každý krok je vysvětlitelný a laditelný
- ✅ **Robustnost**: Funguje konzistentně bez potřeby velkých trénovacích dat

---

## Fáze 3: Inteligentní Kompozice PDF s Automatickým Centrováním (`03_generate_pdf_centering.py`)

### Koncepce
Třetí fáze implementuje **sophisticated layout engine** s automatickou detekcí obsahu a inteligentním umístěním. Systém nejenže vytváří PDF, ale také **automaticky detekuje střed obsahu** a optimalizuje pozicování pro profesionální vzhled.

### Technické Detaily

#### 3.1 Detekce Středu Obsahu (Content Center Detection)
**Toto je pokročilý algoritmus pro automatické rozpoznání rozložení obsahu:**

```python
def find_content_center(img: Image.Image, debug_name: str = None):
```

**Multi-stage proces:**

1. **Gaussovské rozmazání (radius=10)**:
   - Eliminuje drobné artefakty a šum
   - Vytváří hladký gradient pro lepší detekci oblastí

2. **Adaptivní thresholding (threshold=240)**:
   - Binární klasifikace: obsah vs. pozadí
   - Pro bílé pozadí: pixely tmavší než 240 = obsah

3. **Bounding box výpočet**:
   - **Numpy vectorizované operace**: `np.any(mask, axis=1)` a `np.any(mask, axis=0)`
   - **Efektivní algoritmus**: O(n) složitost místo O(n²) při naivním přístupu
   - Najde minimální obdélník obsahující veškerý detekovaný obsah

4. **Centroid výpočet**:
   - Geometrický střed bounding boxu = střed obsahu
   - Používá se pro následné centrování na stránce

**Výsledek**: Automatická detekce, kde se nachází skutečný obsah, bez ohledu na jeho pozici v původním obrázku.

#### 3.2 Vysokokvalitní Resampling
- **LANCZOS resampling**: Používá 3. řád polynomiální interpolaci
- **Sub-pixelová přesnost**: Zachovává detaily lépe než bilineární nebo nearest-neighbor
- **Aspect ratio preservation**: Matematicky přesné zachování poměru stran

**Matematický základ**: LANCZOS používá sinc funkci s okénkováním pro minimalizaci aliasing artefaktů.

#### 3.3 Inteligentní Pozicování s Offsetem
**Systém implementuje sofistikovanou logiku pro knihařské potřeby:**

- **Parita stránek**: Liché stránky mají offset +7mm doprava, sudé -7mm doleva
- **Kompenzace vazby**: Umožňuje prostor pro vazbu při tištění oboustranně
- **Precizní výpočty**: Konverze mm→pixely s DPI=300 pro tiskovou kvalitu

**Výpočet pozice:**
```
page_center = PAGE_WIDTH_PX // 2
content_center_resized = content_center_original * scale_factor
x_position = page_center - content_center_resized + offset_px
```

#### 3.4 Automatické Číslování Stránek
- **Adaptivní font loading**: Pokus o systémové fonty s fallback mechanismem
- **Precizní typografie**: Výpočet bounding boxu textu pro přesné centrování
- **Vizuální optimalizace**: 30% brightness (RGB 77,77,77) pro jemné, nevtíravé číslování

#### 3.5 PDF Generování s Maximální Kvalitou
- **300 DPI**: Profesionální tisková kvalita (vs. standardní 72 DPI pro obrazovku)
- **Lossless komprese**: PNG jako mezikrok zajišťuje bezztrátovou kvalitu
- **Multi-page PDF**: Efektivní spojení všech stránek do jednoho dokumentu

**Výpočet rozlišení:**
```
A4 210×297mm @ 300 DPI = 2480×3508 pixelů
```

---

## Celkový Workflow

```
input_original/ (chaotické soubory)
    ↓ [01_sort.py: Lexikografické řazení + normalizace]
input_sorted/ (page001a.jpg, page002a.jpg, ...)
    ↓ [02_clean_bg.py: Adaptivní thresholding + morfologie + komponentová analýza]
output_cleaned/ (page001a_clean.png, ...)
    ↓ [03_generate_pdf_centering.py: Detekce obsahu + inteligentní layout + PDF generování]
output.pdf (profesionální multi-page dokument)
```

---

## Technologický Stack

- **OpenCV (cv2)**: Průmyslový standard pro počítačové vidění
- **NumPy**: Vektorizované matematické operace pro výkon
- **PIL/Pillow**: Pokročilá manipulace s obrázky a PDF generování
- **Python Pathlib**: Moderní, cross-platform file handling

---

## Výkonnostní Charakteristiky

- **Determinismus**: Žádná randomizace, reprodukovatelné výsledky
- **Efektivita**: O(n) až O(n log n) složitost v závislosti na fázi
- **Škálovatelnost**: Lineární škálování s počtem obrázků
- **Kvalita**: Výsledky srovnatelné s komerčními OCR systémy

---

## Závěr

Tento systém představuje elegantní kombinaci **klasických algoritmů počítačového vidění**, **matematické morfologie** a **inteligentního layout engine**. Na rozdíl od "černých skříněk" neuronových sítí, každý krok je transparentní, laditelný a založený na pevných matematických základech. Výsledek je profesionální digitální dokument, který zachovává všechny detaily originálu při optimalizaci pro tisk a čitelnost.
